'''
Paper: CCDG — Contrastive Condition Domain Generalization for Time Series Data.
    Built on the DomainBed framework (github.com/facebookresearch/DomainBed).
Reference code: https://github.com/mohamedr002/CCDG
Note: Adapted for unsupervised domain adaptation (UDA) in 1D vibration signal
    bearing fault diagnosis. The original CCDG targets domain generalization (DG)
    with multiple labeled domains. Here we adapt it to the single-source UDA setting:
      - Source (labeled):   CE loss + superived contrastive loss (class-wise pull)
      - Target (unlabeled): domain adversarial alignment + conditional entropy minimization

    Core innovations retained from CCDG:
      1. Supervised domain contrastive loss — L2-normalized features,
         temperature-scaled cosine similarity, class-label masking for positive pairs
      2. Same architecture pattern as ERM_Contrastive: shared F + linear C
      3. Augmented with DANN-style domain adversarial training for UDA
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import logging
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

import utils
import model_base
from train_utils import InitTrain


# ======================================================================
# 1. Domain Contrastive Loss (from CCDG lib/misc.py)
# ======================================================================
def domain_contrastive_loss(features, labels, temperature, device):
    """
    Supervised contrastive loss: pulls features of the same class together,
    pushes features of different classes apart.

    Args:
        features: (B, D) — feature vectors
        labels:   (B,)   — class labels
        temperature: scalar — temperature scaling
        device: torch device

    Returns:
        scalar loss

    Reference: CCDG/lib/misc.py → domain_contrastive_loss()
    """
    # L2 normalize features
    anchor_feature = F.normalize(features, dim=1)
    labels = labels.contiguous().view(-1, 1)

    # Build positive/negative mask: mask[i,j]=1 if same class
    mask = torch.eq(labels, labels.T).float().to(device)

    # Cosine similarity matrix (since features are L2-normalized)
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, anchor_feature.T), temperature
    )

    # Numerical stability: subtract max
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # Mask out self-contrast (diagonal = same sample i=j)
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(anchor_feature.shape[0]).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    # Compute log probability
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # Mean over positive pairs
    mask_sum = mask.sum(1)
    zeros_idx = torch.where(mask_sum == 0)[0]
    mask_sum[zeros_idx] = 1  # avoid division by zero

    mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

    # Loss
    loss = -1.0 * mean_log_prob_pos
    loss = loss.mean()

    return loss


# ======================================================================
# 2. Conditional Entropy Loss (target entropy minimization)
# ======================================================================
class ConditionalEntropyLoss(nn.Module):
    """Minimize H(p) = -sum(p * log(p)) to encourage confident target predictions."""
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)


# ======================================================================
# 3. Trainset — Standard Interface (compatible with train.py)
# ======================================================================
class Trainset(InitTrain):
    """
    CCDG: Contrastive Condition Domain Generalization adapted for UDA.

    Components:
      1. Feature extractor F (multi-scale 1D-CNN, from model_base)
      2. Classifier C (linear layer)
      3. Domain adversarial alignment (GRL + discriminator, DANN-style)
      4. Supervised contrastive loss on source features (CCDG core)
      5. Conditional entropy minimization on target predictions

    Compatible with the standard train.py interface:
        python train.py --model_name CCDG --source CWRU_0 --target CWRU_1 --train_mode single_source --cuda_device 0
    """

    def __init__(self, args):
        super(Trainset, self).__init__(args)

        output_size = 2560

        # ---------- 1. Feature Extractor F ----------
        self.F = model_base.FeatureExtractor(
            in_channel=1, dropout=args.dropout
        ).to(self.device)

        # ---------- 2. Classifier C ----------
        self.C = nn.Linear(output_size, args.num_classes).to(self.device)

        # ---------- 3. Domain Discriminator + Adversarial Loss ----------
        self.train_mode = args.train_mode
        grl = utils.GradientReverseLayer()
        if self.train_mode == 'multi_source':
            self.domain_discri = nn.ModuleList([
                model_base.ClassifierMLP(
                    input_size=output_size,
                    output_size=1,
                    dropout=args.dropout,
                    last='sigmoid'
                ) for _ in range(self.num_source)
            ]).to(self.device)
            self.domain_adv = nn.ModuleList([
                utils.DomainAdversarialLoss(d, grl=grl) for d in self.domain_discri
            ])
            self.domain_weight_module = utils.AutomaticUpdateDomainWeightModule(
                num_domains=self.num_source, eta=1e-2, device=self.device
            )
        else:
            self.domain_discri = model_base.ClassifierMLP(
                input_size=output_size,
                output_size=1,
                dropout=args.dropout,
                last='sigmoid'
            ).to(self.device)
            self.domain_adv = utils.DomainAdversarialLoss(self.domain_discri, grl=grl)
            self.domain_weight_module = None

        # ---------- 4. Loss Functions ----------
        self.cross_entropy = nn.CrossEntropyLoss()
        self.entropy_loss = ConditionalEntropyLoss()

        # ---------- 5. Hyperparameters ----------
        self.contrastive_temp = 0.07      # Temperature τ for contrastive loss
        self.contrastive_weight = 0.1     # α: weight for contrastive loss
        self.entropy_weight = 0.1         # β: weight for target entropy minimization

        # ---------- F1 Settings ----------
        self.num_classes = args.num_classes
        self.f1_average = self._get_f1_average()
        logging.info(f"Task: {self.num_classes}-class classification | F1 average: '{self.f1_average}'")

        # ---------- Load Data ----------
        self._init_data()

    def _get_f1_average(self):
        if self.num_classes == 2:
            return 'binary'
        elif self.num_classes > 2:
            return 'weighted'
        else:
            return 'micro'

    # ================== Model Save / Load ==================
    def _discriminator_state_dict(self):
        if self.train_mode == 'multi_source':
            return {f'disc_{i}': d.state_dict() for i, d in enumerate(self.domain_discri)}
        return self.domain_discri.state_dict()

    def save_model_acc(self):
        save_path = self.args.save_path + '_best_acc.pth'
        torch.save({
            'F': self.F.state_dict(),
            'C': self.C.state_dict(),
            'domain_discri': self._discriminator_state_dict(),
        }, save_path)
        logging.info('ACC Model saved to {}'.format(save_path))

    def save_model_f1(self):
        save_path = self.args.save_path + '_best_f1.pth'
        torch.save({
            'F': self.F.state_dict(),
            'C': self.C.state_dict(),
            'domain_discri': self._discriminator_state_dict(),
        }, save_path)
        logging.info('F1 Model saved to {}'.format(save_path))

    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path, map_location=self.device)
        self.F.load_state_dict(ckpt['F'])
        self.C.load_state_dict(ckpt['C'])
        if 'domain_discri' in ckpt:
            if self.train_mode == 'multi_source':
                for i, d in enumerate(self.domain_discri):
                    d.load_state_dict(ckpt['domain_discri'][f'disc_{i}'])
            else:
                self.domain_discri.load_state_dict(ckpt['domain_discri'])

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

        self.optimizer = self._get_optimizer([self.F, self.C, self.domain_discri])
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)

        best_acc = 0.0
        best_acc_epoch = 0
        best_acc_f1 = 0.0

        best_f1 = 0.0
        best_f1_epoch = 0
        best_f1_acc = 0.0

        for epoch in range(1, args.max_epoch + 1):
            self.current_epoch = epoch
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-' * 5)

            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))

            self.F.train()
            self.C.train()
            self.domain_discri.train()

            epoch_loss = defaultdict(float)
            epoch_acc = defaultdict(float)
            tradeoff = self._get_tradeoff(args.tradeoff, epoch)

            num_iter = len(self.dataloaders['train'])
            if num_iter == 0:
                logging.error("Training dataloader is empty!")
                break

            for i in tqdm(range(num_iter), ascii=True):
                self.optimizer.zero_grad()

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

                    f_t = self.F(target_data)
                    logits_t = self.C(f_t)
                    loss_ent = self.entropy_loss(logits_t)

                    loss_per_domain = torch.zeros(self.num_source).to(self.device)
                    loss_adapt_per_domain = torch.zeros(self.num_source).to(self.device)
                    for idx in range(self.num_source):
                        f_s = self.F(source_data[idx])
                        logits_s = self.C(f_s)
                        loss_cls = self.cross_entropy(logits_s, source_labels[idx])
                        epoch_acc['Source'] += utils.get_accuracy(logits_s, source_labels[idx]) / self.num_source

                        loss_cont = domain_contrastive_loss(
                            f_s, source_labels[idx], self.contrastive_temp, self.device
                        )
                        loss_adv, acc_d = self.domain_adv[idx](f_s, f_t)
                        epoch_acc['Discriminator'] += acc_d / self.num_source

                        loss_per_domain[idx] = loss_cls + self.contrastive_weight * loss_cont
                        loss_adapt_per_domain[idx] = loss_adv + self.entropy_weight * loss_ent

                    self.domain_weight_module.update(loss_per_domain, domain_idxes)
                    domain_weight = self.domain_weight_module.get_domain_weight(domain_idxes)
                    loss = (domain_weight * (loss_per_domain + tradeoff[0] * loss_adapt_per_domain)).sum()

                    loss.backward()
                    self.optimizer.step()

                    epoch_loss['Source Classifier'] += (domain_weight * loss_per_domain).sum().item()
                    epoch_loss['Discriminator'] += (domain_weight * loss_adapt_per_domain).sum().item()
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

                # ===== Forward: Shared Feature Extractor =====
                f_s = self.F(source_data)
                f_t = self.F(target_data)

                # ===== Source Classification Loss =====
                logits_s = self.C(f_s)
                loss_cls = self.cross_entropy(logits_s, source_labels)
                epoch_acc['Source'] += utils.get_accuracy(logits_s, source_labels)

                # ===== CCDG: Supervised Contrastive Loss (source only) =====
                loss_cont = domain_contrastive_loss(
                    f_s, source_labels, self.contrastive_temp, self.device
                )

                # ===== Domain Adversarial Loss =====
                loss_adv, acc_d = self.domain_adv(f_s, f_t)
                epoch_acc['Discriminator'] += acc_d

                # ===== Target Entropy Minimization =====
                logits_t = self.C(f_t)
                loss_ent = self.entropy_loss(logits_t)

                # ===== Total Loss =====
                loss = loss_cls \
                     + self.contrastive_weight * loss_cont \
                     + tradeoff[0] * loss_adv \
                     + self.entropy_weight * loss_ent

                loss.backward()
                self.optimizer.step()

                epoch_loss['Source Classifier'] += loss_cls.item()
                epoch_loss['Contrastive'] += loss_cont.item()
                epoch_loss['Discriminator'] += loss_adv.item()
                epoch_loss['Target Entropy'] += loss_ent.item()

            # ---- Print training stats ----
            for key in sorted(epoch_loss.keys()):
                logging.info('Train-Loss {}: {:.4f}'.format(
                    key, epoch_loss[key] / max(num_iter, 1)))
            for key in sorted(epoch_acc.keys()):
                logging.info('Train-Acc {}: {:.4f}'.format(
                    key, epoch_acc[key] / max(num_iter, 1)))

            # ---- Validation ----
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

            logging.info(f"Epoch {epoch}: Val-Acc={new_acc:.4f}, Val-F1={new_f1:.4f}")
            logging.info("Best so far -> "
                         f"Acc: Epoch {best_acc_epoch} (Acc={best_acc:.4f}, F1={best_acc_f1:.4f}) | "
                         f"F1: Epoch {best_f1_epoch} (Acc={best_f1_acc:.4f}, F1={best_f1:.4f})")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        logging.info('=' * 70)
        logging.info('Training finished! Best results summary:')
        logging.info(f'Best Accuracy: {best_acc:.4f} (Epoch {best_acc_epoch}, F1={best_acc_f1:.4f})')
        logging.info(f'Best F1-score: {best_f1:.4f} (Epoch {best_f1_epoch}, Acc={best_f1_acc:.4f})')
        logging.info('=' * 70)

    # ================== Validation ==================
    @torch.no_grad()
    def test(self):
        self.F.eval()
        self.C.eval()
        self.domain_discri.eval()

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

            f_t = self.F(target_data)
            logits = self.C(f_t)
            pred_labels = torch.argmax(logits, dim=1)

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
            logging.warning(f"Only {len(unique_labels)} class(es) present. F1 set to 0.")
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
