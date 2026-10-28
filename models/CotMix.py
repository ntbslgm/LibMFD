'''
Paper: CoTMix - Temporal Mixup with Contrastive Learning and Domain Adversarial Training for Fault Diagnosis
Note: This implementation adapts the CoTMix algorithm to the standard TL-Bearing-Fault-Diagnosis training interface.
Improved version with feature-level mixing and category-aware strategies.
Fixed version without register_buffer dependency.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import logging
import hashlib
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

import utils
import model_base
from train_utils import InitTrain


# ======================================================================
# 1. Loss Functions
# ======================================================================
class ConditionalEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)


class NTXentLoss(torch.nn.Module):
    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_similarity_func
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        return v

    def _cosine_similarity_func(self, x, y):
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)
        similarity_matrix = self.similarity_function(representations, representations)

        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


class SupConLoss(torch.nn.Module):
    def __init__(self, device, temperature=0.2, contrast_mode='all'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.device = device

    def forward(self, features, labels=None, mask=None):
        device = self.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature,
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - self.temperature * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


# ======================================================================
# 2. Domain Discriminator
# ======================================================================
class DiscriminatorMLP(nn.Module):
    def __init__(self, input_size, output_size=2, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, input_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_size // 4, input_size // 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_size // 16, output_size),
        )

    def forward(self, x):
        return self.net(x)


# ======================================================================
# 3. Gradient Reversal Layer
# ======================================================================
class GradientReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=1.0):
    return GradientReverseLayer.apply(x, alpha)


# ======================================================================
# 4. Projection Head for Contrastive Learning
# ======================================================================
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# ======================================================================
# 5. Trainset — Fixed Version
# ======================================================================
class Trainset(InitTrain):
    """
    Improved CoTMix: Feature-level Mixup + Category-aware Strategies + Contrastive Learning.
    Fixed version without register_buffer dependency.
    """

    def __init__(self, args):
        super(Trainset, self).__init__(args)

        # ---------- Task-specific random seed ----------
        task_seed = self._get_task_seed(args)
        torch.manual_seed(task_seed)
        np.random.seed(task_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(task_seed)
        logging.info(f'Task seed: {task_seed} (source={args.source_name}, target={args.target})')

        # ---------- Dynamically determine feature dimension ----------
        with torch.no_grad():
            dummy_extractor = model_base.FeatureExtractor(
                in_channel=1, dropout=args.dropout
            )
            dummy_input = torch.randn(1, 1, args.signal_size)
            self.feature_dim = dummy_extractor(dummy_input).shape[1]
            del dummy_extractor
        logging.info(f'Inferred feature dimension: {self.feature_dim}')

        # ---------- Feature Extractor ----------
        self.feature_extractor = model_base.FeatureExtractor(
            in_channel=1, dropout=args.dropout
        ).to(self.device)

        # ---------- Classifier ----------
        self.classifier = nn.Linear(self.feature_dim, args.num_classes).to(self.device)

        # ---------- Projection Head for Contrastive Learning ----------
        self.projection_head = ProjectionHead(
            input_dim=self.feature_dim,
            hidden_dim=128,
            output_dim=64
        ).to(self.device)

        # ---------- Domain Discriminator ----------
        discriminator_input_size = self.feature_dim + args.num_classes
        self.discriminator = DiscriminatorMLP(
            input_size=discriminator_input_size,
            output_size=2,
            dropout=args.dropout,
        ).to(self.device)

        # ---------- Category Prototype Memory Bank (Fixed: Use nn.Parameter) ----------
        self.class_prototypes = nn.Parameter(
            torch.randn(args.num_classes, self.feature_dim, device=self.device),
            requires_grad=False
        )
        self.prototype_update_rate = 0.1

        # ---------- Explicit weight initialization ----------
        self._init_weights()

        # ---------- Beta Distribution for feature mixup ----------
        self.dist_beta = torch.distributions.beta.Beta(0.2, 0.2)

        # ---------- Loss Functions ----------
        self.cross_entropy = nn.CrossEntropyLoss()
        self.contrastive_loss = NTXentLoss(self.device, args.batch_size, 0.2, True)
        self.entropy_loss = ConditionalEntropyLoss()
        self.sup_contrastive_loss = SupConLoss(self.device)

        # ---------- Adjusted Hyperparameters ----------
        self.mix_ratio = 0.3
        self.temporal_shift = 1
        self.src_cls_weight = 1.0
        self.trg_entropy_weight = 0.05
        self.src_supCon_weight = 0.3
        self.trg_cont_weight = 0.1
        self.discriminator_weight = 0.05
        self.separation_weight = 0.01

        # ---------- GRL warmup settings ----------
        self.grl_alpha_min = 0.0
        self.grl_alpha_max = 1.0
        self.grl_warmup_epochs = max(1, args.max_epoch // 5)

        # ---------- Gradient clipping ----------
        self.grad_clip_value = 5.0

        # ---------- F1 Settings ----------
        self.num_classes = args.num_classes
        self.f1_average = self._get_f1_average()
        logging.info(f"Task: {self.num_classes}-class classification | F1 average: '{self.f1_average}'")

        # ---------- Multi-source support ----------
        self.train_mode = args.train_mode
        self.domain_weight_module = utils.AutomaticUpdateDomainWeightModule(
            num_domains=self.num_source, eta=1e-2, device=self.device
        ) if self.train_mode == 'multi_source' else None

        # ---------- Load Data ----------
        self._init_data()

    # ================== Helper Methods ==================

    def _get_task_seed(self, args):
        src_str = '_'.join(sorted(args.source_name))
        task_id = f"{src_str}__{args.target}"
        hash_int = int(hashlib.md5(task_id.encode()).hexdigest(), 16) % (2**30)
        return hash_int

    def _init_weights(self):
        """Apply Kaiming initialization for better gradient flow."""
        # Feature extractor
        for m in self.feature_extractor.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Classifier: Xavier init
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

        # Projection head: Xavier init
        for m in self.projection_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Discriminator: Kaiming init for hidden layers, Xavier for output
        for m in self.discriminator.modules():
            if isinstance(m, nn.Linear):
                if m.out_features == 2:  # Output layer
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:  # Hidden layers
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        # Initialize class prototypes
        nn.init.xavier_uniform_(self.class_prototypes)

    def _reset_iterators(self):
        dataset_keys = self.args.source_name + ['train', 'val']
        if hasattr(self, 'datasets') and 'concat_source' in self.datasets:
            dataset_keys.append('concat_source')
        if hasattr(self, 'datasets') and 'concat_all' in self.datasets:
            dataset_keys.append('concat_all')
        self.iters = {x: iter(self.dataloaders[x]) for x in dataset_keys if x in self.dataloaders}

    def _get_grl_alpha(self, epoch):
        if epoch >= self.grl_warmup_epochs:
            return self.grl_alpha_max
        progress = epoch / max(self.grl_warmup_epochs, 1)
        return self.grl_alpha_min + (self.grl_alpha_max - self.grl_alpha_min) * progress

    def _get_f1_average(self):
        if self.num_classes == 2:
            return 'binary'
        elif self.num_classes > 2:
            return 'weighted'
        else:
            return 'micro'

    # ================== Model Save / Load ==================
    def save_model_acc(self):
        save_path = self.args.save_path + '_best_acc.pth'
        torch.save({
            'feature_extractor': self.feature_extractor.state_dict(),
            'classifier': self.classifier.state_dict(),
            'projection_head': self.projection_head.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'class_prototypes': self.class_prototypes.data.cpu(),
        }, save_path)
        logging.info('ACC Model saved to {}'.format(save_path))

    def save_model_f1(self):
        save_path = self.args.save_path + '_best_f1.pth'
        torch.save({
            'feature_extractor': self.feature_extractor.state_dict(),
            'classifier': self.classifier.state_dict(),
            'projection_head': self.projection_head.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'class_prototypes': self.class_prototypes.data.cpu(),
        }, save_path)
        logging.info('F1 Model saved to {}'.format(save_path))

    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path, map_location=self.device)
        self.feature_extractor.load_state_dict(ckpt['feature_extractor'])
        self.classifier.load_state_dict(ckpt['classifier'])
        if 'projection_head' in ckpt:
            self.projection_head.load_state_dict(ckpt['projection_head'])
        if 'discriminator' in ckpt:
            self.discriminator.load_state_dict(ckpt['discriminator'])
        if 'class_prototypes' in ckpt:
            self.class_prototypes.data.copy_(ckpt['class_prototypes'].to(self.device))

    # ================== Training ==================
    def train(self):
        args = self.args

        if args.train_mode == 'single_source':
            src = args.source_name[0]
        elif args.train_mode == 'source_combine':
            src = args.source_name
        elif args.train_mode == 'multi_source':
            src = args.source_name
            domain_idxes = list(range(self.num_source))

        # Separate optimizers
        self.optimizer_fe = torch.optim.Adam(
            self.feature_extractor.parameters(),
            lr=args.lr,
            weight_decay=1e-4
        )
        self.optimizer_cls = torch.optim.Adam(
            self.classifier.parameters(),
            lr=args.lr * 2,
            weight_decay=1e-4
        )
        self.optimizer_proj = torch.optim.Adam(
            self.projection_head.parameters(),
            lr=args.lr * 0.5,
            weight_decay=1e-4
        )
        self.optimizer_dm = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=args.lr * 0.1,
            weight_decay=1e-4
        )

        # Learning rate schedulers
        self.scheduler_fe = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_fe, T_max=args.max_epoch
        )
        self.scheduler_cls = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_cls, T_max=args.max_epoch
        )
        self.scheduler_proj = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_proj, T_max=args.max_epoch
        )
        self.scheduler_dm = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_dm, T_max=args.max_epoch
        )

        best_acc = 0.0
        best_acc_epoch = 0
        best_acc_f1 = 0.0

        best_f1 = 0.0
        best_f1_epoch = 0
        best_f1_acc = 0.0

        for epoch in range(1, args.max_epoch + 1):
            self.current_epoch = epoch
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-' * 5)

            logging.info('Learning rates:')
            logging.info(f'  Feature Extractor: {self.scheduler_fe.get_last_lr()[0]:.6f}')
            logging.info(f'  Classifier: {self.scheduler_cls.get_last_lr()[0]:.6f}')
            logging.info(f'  Projection Head: {self.scheduler_proj.get_last_lr()[0]:.6f}')
            logging.info(f'  Discriminator: {self.scheduler_dm.get_last_lr()[0]:.6f}')

            self.feature_extractor.train()
            self.classifier.train()
            self.projection_head.train()
            self.discriminator.train()

            self._reset_iterators()
            grl_alpha = self._get_grl_alpha(epoch)

            epoch_loss = defaultdict(float)
            num_iter = len(self.dataloaders['train'])

            if num_iter == 0:
                logging.error("Training dataloader is empty!")
                break

            for i in tqdm(range(num_iter), ascii=True):
                self.optimizer_fe.zero_grad()
                self.optimizer_cls.zero_grad()
                self.optimizer_proj.zero_grad()
                self.optimizer_dm.zero_grad()

                if args.train_mode == 'multi_source':
                    try:
                        target_data, _ = utils.get_next_batch(
                            self.dataloaders, self.iters, 'train', self.device
                        )
                        source_data, source_labels = [], []
                        for idx in range(self.num_source):
                            d, l = utils.get_next_batch(
                                self.dataloaders, self.iters, src[idx], self.device
                            )
                            source_data.append(d)
                            source_labels.append(l)
                    except StopIteration:
                        break

                    loss_per_domain = torch.zeros(self.num_source).to(self.device)
                    total_losses = []
                    for idx in range(self.num_source):
                        loss_dict = self.update(source_data[idx], source_labels[idx], target_data, grl_alpha)
                        loss_per_domain[idx] = loss_dict['src_cls_loss']
                        total_losses.append(loss_dict['total_loss'])

                    self.domain_weight_module.update(loss_per_domain, domain_idxes)
                    domain_weight = self.domain_weight_module.get_domain_weight(domain_idxes)
                    loss = sum(w * l for w, l in zip(domain_weight, total_losses))

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.feature_extractor.parameters(), self.grad_clip_value
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.classifier.parameters(), self.grad_clip_value
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.projection_head.parameters(), self.grad_clip_value
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.discriminator.parameters(), self.grad_clip_value
                    )
                    self.optimizer_fe.step()
                    self.optimizer_cls.step()
                    self.optimizer_proj.step()
                    self.optimizer_dm.step()

                    epoch_loss['src_cls_loss'] += (domain_weight * loss_per_domain).sum().item()
                    continue

                try:
                    target_data, _ = utils.get_next_batch(
                        self.dataloaders, self.iters, 'train', self.device
                    )
                    source_data, source_labels = utils.get_next_batch(
                        self.dataloaders, self.iters, src, self.device
                    )
                except StopIteration:
                    break

                loss_dict = self.update(source_data, source_labels, target_data, grl_alpha)

                loss_dict['total_loss'].backward()

                torch.nn.utils.clip_grad_norm_(
                    self.feature_extractor.parameters(), self.grad_clip_value
                )
                torch.nn.utils.clip_grad_norm_(
                    self.classifier.parameters(), self.grad_clip_value
                )
                torch.nn.utils.clip_grad_norm_(
                    self.projection_head.parameters(), self.grad_clip_value
                )
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(), self.grad_clip_value
                )

                self.optimizer_fe.step()
                self.optimizer_cls.step()
                self.optimizer_proj.step()
                self.optimizer_dm.step()

                for k, v in loss_dict.items():
                    if k != 'total_loss':
                        epoch_loss[k] += v if isinstance(v, float) else v.item()

            for key in sorted(epoch_loss.keys()):
                logging.info('Train-Loss {}: {:.4f}'.format(key, epoch_loss[key] / max(num_iter, 1)))

            new_acc, new_f1 = self.test()

            if new_acc >= best_acc:
                best_acc = new_acc
                best_acc_epoch = epoch
                best_acc_f1 = new_f1
                self.save_model_acc()

            if new_f1 >= best_f1:
                best_f1 = new_f1
                best_f1_epoch = epoch
                best_f1_acc = new_acc
                self.save_model_f1()

            logging.info(f"Epoch {epoch}: Val-Acc={new_acc:.4f}, Val-F1={new_f1:.4f}, GRL-alpha={grl_alpha:.3f}")
            logging.info("Best so far -> "
                         f"Acc: Epoch {best_acc_epoch} (Acc={best_acc:.4f}, F1={best_acc_f1:.4f}) | "
                         f"F1: Epoch {best_f1_epoch} (Acc={best_f1_acc:.4f}, F1={best_f1:.4f})")

            self.scheduler_fe.step()
            self.scheduler_cls.step()
            self.scheduler_proj.step()
            self.scheduler_dm.step()

        logging.info('=' * 70)
        logging.info('Training finished! Best results summary:')
        logging.info(f'Best Accuracy: {best_acc:.4f} (Epoch {best_acc_epoch}, F1={best_acc_f1:.4f})')
        logging.info(f'Best F1-score: {best_f1:.4f} (Epoch {best_f1_epoch}, Acc={best_f1_acc:.4f})')
        logging.info('=' * 70)

    # ================== Core Update Function ==================
    def update(self, src_x, src_y, trg_x, grl_alpha=1.0):
        batch_size = src_x.shape[0]

        # ----- 1. Extract features first -----
        src_feat = self.feature_extractor(src_x)
        trg_feat = self.feature_extractor(trg_x)

        # ----- 2. Update class prototypes using source features -----
        with torch.no_grad():
            for cls_idx in range(self.num_classes):
                cls_mask = (src_y == cls_idx)
                if cls_mask.any():
                    cls_feat = src_feat[cls_mask].mean(dim=0)
                    # EMA update - use .data to avoid tracking in computation graph
                    self.class_prototypes.data[cls_idx] = (
                        (1 - self.prototype_update_rate) * self.class_prototypes.data[cls_idx] +
                        self.prototype_update_rate * cls_feat
                    )

        # ----- 3. Category-aware feature mixing for source domain -----
        mixed_src_feat = src_feat.clone()

        for cls_idx in range(self.num_classes):
            cls_mask = (src_y == cls_idx)
            if cls_mask.sum() > 1:
                cls_indices = torch.where(cls_mask)[0]
                perm = torch.randperm(len(cls_indices))

                lam = self.dist_beta.sample((len(cls_indices), 1)).to(self.device)
                mixed_src_feat[cls_indices] = (
                    lam * src_feat[cls_indices] +
                    (1 - lam) * src_feat[cls_indices[perm]]
                )

        # ----- 4. Target domain mixing with class prototypes -----
        mixed_trg_feat = trg_feat.clone()

        with torch.no_grad():
            trg_proto_dist = torch.cdist(trg_feat, self.class_prototypes.data)
            nearest_proto_idx = trg_proto_dist.argmin(dim=1)

            lam = self.dist_beta.sample((batch_size, 1)).to(self.device)
            mixed_trg_feat = (
                lam * trg_feat +
                (1 - lam) * self.class_prototypes.data[nearest_proto_idx]
            )

        # ----- 5. Get logits -----
        src_orig_logits = self.classifier(src_feat)
        trg_orig_logits = self.classifier(trg_feat)

        src_mixed_logits = self.classifier(mixed_src_feat)
        trg_mixed_logits = self.classifier(mixed_trg_feat)

        # ----- 6. Loss calculations -----
        src_cls_loss = self.cross_entropy(src_orig_logits, src_y)
        total_loss = src_cls_loss * self.src_cls_weight

        trg_entropy_loss = self.entropy_loss(trg_orig_logits)
        total_loss += trg_entropy_loss * self.trg_entropy_weight

        src_proj = self.projection_head(mixed_src_feat)
        src_proj = F.normalize(src_proj, dim=1)

        src_orig_proj = self.projection_head(src_feat)
        src_orig_proj = F.normalize(src_orig_proj, dim=1)

        src_views = torch.cat([
            src_orig_proj.unsqueeze(1),
            src_proj.unsqueeze(1)
        ], dim=1)

        src_supcon_loss = self.sup_contrastive_loss(src_views, src_y)
        total_loss += src_supcon_loss * self.src_supCon_weight

        trg_proj = self.projection_head(trg_feat)
        trg_proj = F.normalize(trg_proj, dim=1)

        noise = torch.randn_like(trg_proj) * 0.1
        trg_aug_proj = F.normalize(trg_proj + noise, dim=1)

        trg_con_loss = self.contrastive_loss(trg_proj, trg_aug_proj)
        total_loss += trg_con_loss * self.trg_cont_weight

        separation_loss = torch.tensor(0.0, device=self.device)
        if self.current_epoch > 5:
            proto_dist = torch.cdist(self.class_prototypes.data, self.class_prototypes.data)
            mask = 1 - torch.eye(self.num_classes, device=self.device)
            separation_loss = -(proto_dist * mask).mean()
            total_loss += separation_loss * self.separation_weight

        # ----- 7. Improved domain discriminator -----
        with torch.no_grad():
            src_feat_norm = F.normalize(src_feat, dim=1)
            trg_feat_norm = F.normalize(trg_feat, dim=1)

            src_softmax = F.softmax(src_orig_logits, dim=1)
            trg_softmax = F.softmax(trg_orig_logits, dim=1)

            src_repr = torch.cat([src_feat_norm, src_softmax], dim=1)
            trg_repr = torch.cat([trg_feat_norm, trg_softmax], dim=1)

        lam = torch.rand(batch_size, 1, device=self.device)
        mixed_src_repr = lam * src_repr + (1 - lam) * src_repr[torch.randperm(batch_size)]
        mixed_trg_repr = lam * trg_repr + (1 - lam) * trg_repr[torch.randperm(batch_size)]

        domain_input = grad_reverse(
            torch.cat([mixed_src_repr, mixed_trg_repr], dim=0),
            alpha=grl_alpha
        )

        domain_logits = self.discriminator(domain_input)

        domain_labels = torch.cat([
            torch.ones(batch_size, device=self.device),
            torch.zeros(batch_size, device=self.device)
        ]).long()

        domain_loss = self.cross_entropy(domain_logits, domain_labels)
        total_loss += domain_loss * self.discriminator_weight

        with torch.no_grad():
            grad_norm_fe = self._get_grad_norm(self.feature_extractor)
            grad_norm_cls = self._get_grad_norm(self.classifier)
            grad_norm_proj = self._get_grad_norm(self.projection_head)
            grad_norm_dm = self._get_grad_norm(self.discriminator)

        return {
            'total_loss': total_loss,
            'src_cls_loss': src_cls_loss.item(),
            'trg_entropy_loss': trg_entropy_loss.item(),
            'src_supcon_loss': src_supcon_loss.item(),
            'trg_con_loss': trg_con_loss.item(),
            'separation_loss': separation_loss.item(),
            'domain_loss': domain_loss.item(),
            'src_acc': (src_orig_logits.argmax(1) == src_y).float().mean().item(),
            'domain_acc': (domain_logits.argmax(1) == domain_labels).float().mean().item(),
            'grad_fe': grad_norm_fe,
            'grad_cls': grad_norm_cls,
            'grad_proj': grad_norm_proj,
            'grad_dm': grad_norm_dm,
        }

    @staticmethod
    def _get_grad_norm(model):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    # ================== Validation ==================
    @torch.no_grad()
    def test(self):
        self.feature_extractor.eval()
        self.classifier.eval()
        self.projection_head.eval()
        self.discriminator.eval()

        all_preds = []
        all_labels = []

        if 'val' not in self.dataloaders or self.dataloaders['val'] is None:
            logging.warning("No validation dataloader found!")
            return 0.0, 0.0

        try:
            iters = iter(self.dataloaders['val'])
            num_iter = len(self.dataloaders['val'])
        except:
            logging.warning("Failed to create validation iterator!")
            return 0.0, 0.0

        if num_iter == 0:
            logging.warning("Validation dataloader is empty!")
            return 0.0, 0.0

        for i in range(num_iter):
            try:
                batch = next(iters)
                if len(batch) == 2:
                    target_data, target_labels = batch
                else:
                    target_data, target_labels, _ = batch
            except StopIteration:
                break
            except Exception as e:
                logging.warning(f"Error getting batch: {e}")
                continue

            if target_data is None or target_labels is None:
                continue

            target_data = target_data.to(self.device)
            target_labels = target_labels.to(self.device)

            feat = self.feature_extractor(target_data)
            pred_logits = self.classifier(feat)
            pred_labels = torch.argmax(pred_logits, dim=1)

            all_preds.extend(pred_labels.cpu().numpy())
            all_labels.extend(target_labels.cpu().numpy())

        if len(all_preds) == 0:
            logging.warning("No predictions made during validation!")
            return 0.0, 0.0

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        acc = accuracy_score(all_labels, all_preds)

        unique_labels = np.unique(all_labels)
        if len(unique_labels) < 2:
            f1 = 0.0
            logging.warning(f"Only {len(unique_labels)} class(es) present in validation set. F1 score is not defined.")
        else:
            try:
                if self.num_classes == 2:
                    pos_label = 1 if 1 in unique_labels else unique_labels[-1]
                    f1 = f1_score(all_labels, all_preds, average='binary', pos_label=pos_label)
                else:
                    f1 = f1_score(all_labels, all_preds, average=self.f1_average)
            except Exception as e:
                logging.warning(f"Error calculating F1: {e}")
                f1 = f1_score(all_labels, all_preds, average='micro')

        logging.info('Val-Acc : {:.4f}'.format(acc))
        logging.info('Val-F1  : {:.4f}'.format(f1))

        return acc, f1