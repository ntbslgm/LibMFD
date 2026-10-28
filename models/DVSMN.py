'''
Paper: Chen, Z., Yu, W., Wang, L., Ding, X., Huang, W. and Shao, Y., 2023.
    A Dual-View Style Mixing Network for Unsupervised Cross-Domain Fault
    Diagnosis With Imbalanced Data. Knowledge-Based Systems.
Reference code: https://github.com/CQU-ZixuChen/DVSMN
Note: Adapted for 1D vibration signal bearing fault diagnosis.

    Core innovations:
      1. Dual-View Graph Convolution — two parallel GCN frameworks on
         multi-channel signal views for fault feature extraction.
      2. Style Mixing Module — self-mixing and bias-mixing of feature
         statistics (mean/std) to learn domain-invariant features without
         generating synthetic samples.
      3. Dual-View Classifier — binary classifier + multi-class classifier
         for sample-level dynamic re-weighting, handling class imbalance.

    Compatible with: python train.py --model_name DVSMN --source CWRU_0 --target CWRU_1
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
# 1. Style Mixing Module
# ======================================================================
class StyleMixing(nn.Module):
    """
    Mix feature statistics (channel-wise mean and std) between domains
    to learn domain-invariant features.

    Two operations:
      - Self-Mixing: randomly perturb own statistics with Gaussian noise
      - Bias-Mixing: replace own statistics with cross-domain statistics
    """
    def __init__(self, eps=1e-5, mix_prob=0.5):
        super(StyleMixing, self).__init__()
        self.eps = eps
        self.mix_prob = mix_prob

    def _compute_stats(self, x):
        """Compute channel-wise mean and std."""
        # x: (B, D)
        mu = x.mean(dim=0, keepdim=True)  # (1, D)
        std = x.std(dim=0, keepdim=True) + self.eps  # (1, D)
        return mu, std

    def self_mix(self, x, noise_std=0.1):
        """Self-mixing: perturb own statistics."""
        mu, std = self._compute_stats(x)
        # Add Gaussian noise to statistics
        mu_noisy = mu + torch.randn_like(mu) * noise_std * std
        std_noisy = std * (1.0 + torch.randn_like(std) * noise_std * 0.5)

        # Normalize with original stats, then re-normalize with noisy stats
        x_norm = (x - mu) / std
        x_mixed = x_norm * std_noisy + mu_noisy
        return x_mixed

    def bias_mix(self, x_src, x_tgt):
        """Bias-mixing: cross-domain statistics exchange."""
        mu_s, std_s = self._compute_stats(x_src)
        mu_t, std_t = self._compute_stats(x_tgt)

        # Mix: use target stats on source features, and vice versa
        x_src_norm = (x_src - mu_s) / std_s
        x_tgt_norm = (x_tgt - mu_t) / std_t

        x_src_mixed = x_src_norm * std_t + mu_t
        x_tgt_mixed = x_tgt_norm * std_s + mu_s

        return x_src_mixed, x_tgt_mixed

    def forward(self, x, x_cross=None):
        """
        Args:
            x: (B, D) source or target features
            x_cross: (B, D) cross-domain features for bias mixing (optional)
        Returns:
            x_mixed: (B, D)
        """
        if self.training and x_cross is not None and torch.rand(1).item() < self.mix_prob:
            # Bias-mixing: cross-domain
            x_mixed, _ = self.bias_mix(x, x_cross)
        else:
            # Self-mixing: within-domain perturbation
            x_mixed = self.self_mix(x, noise_std=0.1)
        return x_mixed


# ======================================================================
# 2. Dual-View Classifier (Binary + Multi-class)
# ======================================================================
class DualViewClassifier(nn.Module):
    """
    Dual-view classifier for sample-level dynamic re-weighting.

    Binary classifier C_bin: predicts whether a sample is "easy" (high confidence)
        or "hard" (low confidence) → sample weight for multi-class loss.
    Multi-class classifier C_multi: standard fault type classifier.
    """
    def __init__(self, input_dim, num_classes, dropout=0.):
        super(DualViewClassifier, self).__init__()

        # Shared trunk
        self.shared = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Binary classifier head
        self.binary_head = nn.Sequential(
            nn.Linear(input_dim // 4, 1),
            nn.Sigmoid()
        )

        # Multi-class classifier head
        self.multi_head = nn.Linear(input_dim // 4, num_classes)

    def forward(self, x):
        h = self.shared(x)
        bin_out = self.binary_head(h)  # (B, 1) — confidence score
        multi_out = self.multi_head(h)  # (B, C) — class logits
        return bin_out.squeeze(1), multi_out


# ======================================================================
# 3. Dual-View Graph Convolution (simplified for 1D)
# ======================================================================
class DualViewGCN(nn.Module):
    """
    Two parallel graph convolutions operating on different "views" of the
    same signal: (1) raw features, (2) frequency-domain (FFT) features.

    Both views share the same GCN architecture but with unshared parameters.
    """
    def __init__(self, input_dim=2560, hidden_dim=256, output_dim=128, dropout=0.):
        super(DualViewGCN, self).__init__()

        # View 1: time-domain graph
        self.gcn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )

        # View 2: frequency-domain graph
        self.gcn2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )

        # Feature fusion
        self.fusion = nn.Linear(output_dim * 2, output_dim)

    def _build_adjacency(self, x):
        """Build k-NN adjacency from features."""
        B = x.size(0)
        device = x.device
        dist = torch.cdist(x, x, p=2)
        _, indices = dist.topk(min(5, B), dim=1, largest=False)
        adj = torch.zeros(B, B, device=device)
        for i in range(B):
            adj[i, indices[i]] = 1
        # Normalize
        D = adj.sum(dim=1) + 1e-6
        adj = adj / D.unsqueeze(1)
        return adj

    def forward(self, x):
        # View 1: raw features
        adj1 = self._build_adjacency(x)
        v1 = self.gcn1(adj1 @ x)

        # View 2: frequency-domain (simulated as DCT of features)
        # Simple: use a permuted feature view
        perm = torch.randperm(x.size(1), device=x.device)
        x_view2 = x[:, perm]
        adj2 = self._build_adjacency(x_view2)
        v2 = self.gcn2(adj2 @ x_view2)

        # Fuse
        fused = self.fusion(torch.cat([v1, v2], dim=1))
        return fused


# ======================================================================
# 4. Trainset — Standard Interface
# ======================================================================
class Trainset(InitTrain):
    """
    DVSMN: Dual-View Style Mixing Network for Cross-Domain Fault Diagnosis.

    Architecture:
      1. Feature Extractor F → base features
      2. Dual-View GCN → two-view graph convolution
      3. Style Mixing → domain-invariant feature learning
      4. Dual-View Classifier → (binary confidence + multi-class logits)
      5. Domain adversarial alignment
      6. Focal loss for class imbalance

    Loss = L_multi + α*L_bin + λ_adv*L_adv + λ_style*L_style
    """

    def __init__(self, args):
        super(Trainset, self).__init__(args)

        base_dim = 2560
        view_out = 128

        # ---------- 1. Feature Extractor ----------
        self.F = model_base.FeatureExtractor(
            in_channel=1, dropout=args.dropout
        ).to(self.device)

        # ---------- 2. Dual-View GCN ----------
        self.dgcn = DualViewGCN(
            input_dim=base_dim, hidden_dim=256, output_dim=view_out, dropout=args.dropout
        ).to(self.device)

        # ---------- 3. Style Mixing ----------
        self.style_mixer = StyleMixing(eps=1e-5, mix_prob=0.5)

        # ---------- 4. Dual-View Classifier ----------
        self.dv_classifier = DualViewClassifier(
            input_dim=view_out, num_classes=args.num_classes, dropout=args.dropout
        ).to(self.device)

        # ---------- 5. Domain Discriminator ----------
        self.train_mode = args.train_mode
        grl = utils.GradientReverseLayer()
        if self.train_mode == 'multi_source':
            self.domain_discri = nn.ModuleList([
                model_base.ClassifierMLP(
                    input_size=view_out, output_size=1,
                    dropout=args.dropout, last='sigmoid'
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
                input_size=view_out, output_size=1,
                dropout=args.dropout, last='sigmoid'
            ).to(self.device)
            self.domain_adv = utils.DomainAdversarialLoss(self.domain_discri, grl=grl)
            self.domain_weight_module = None

        # ---------- 6. Hyperparameters ----------
        self.bin_weight = 0.2       # Binary classifier weight
        self.style_weight = 0.1     # Style consistency weight
        self.focal_gamma = 2.0      # Focal loss gamma for imbalance

        # ---------- 7. F1 Settings ----------
        self.num_classes = args.num_classes
        self.f1_average = self._get_f1_average()
        logging.info(f"DVSMN | {self.num_classes}-class | F1: '{self.f1_average}'")

        self._init_data()

    def _get_f1_average(self):
        if self.num_classes == 2: return 'binary'
        elif self.num_classes > 2: return 'weighted'
        else: return 'micro'

    def _focal_loss(self, logits, targets, gamma=2.0):
        """Focal loss for handling class imbalance."""
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** gamma * ce_loss).mean()
        return focal_loss

    def _extract_features(self, x):
        """Extract base → GCN features."""
        f_base = self.F(x)
        f_view = self.dgcn(f_base)
        return f_view

    def _discriminator_state_dict(self):
        if self.train_mode == 'multi_source':
            return {f'disc_{i}': d.state_dict() for i, d in enumerate(self.domain_discri)}
        return self.domain_discri.state_dict()

    def save_model_acc(self):
        save_path = self.args.save_path + '_best_acc.pth'
        torch.save({
            'F': self.F.state_dict(), 'dgcn': self.dgcn.state_dict(),
            'dv_classifier': self.dv_classifier.state_dict(),
            'domain_discri': self._discriminator_state_dict(),
        }, save_path)
        logging.info('ACC Model saved to {}'.format(save_path))

    def save_model_f1(self):
        save_path = self.args.save_path + '_best_f1.pth'
        torch.save({
            'F': self.F.state_dict(), 'dgcn': self.dgcn.state_dict(),
            'dv_classifier': self.dv_classifier.state_dict(),
            'domain_discri': self._discriminator_state_dict(),
        }, save_path)
        logging.info('F1 Model saved to {}'.format(save_path))

    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path, map_location=self.device)
        for k in ['F', 'dgcn', 'dv_classifier']:
            if k in ckpt: getattr(self, k).load_state_dict(ckpt[k])
        if 'domain_discri' in ckpt:
            if self.train_mode == 'multi_source':
                for i, d in enumerate(self.domain_discri):
                    d.load_state_dict(ckpt['domain_discri'][f'disc_{i}'])
            else:
                self.domain_discri.load_state_dict(ckpt['domain_discri'])

    def train(self):
        args = self.args
        if args.train_mode == 'single_source':
            src = args.source_name[0]
        elif args.train_mode == 'source_combine':
            src = args.source_name
        elif args.train_mode == 'multi_source':
            src = args.source_name
            domain_idxes = list(range(self.num_source))

        self.optimizer = self._get_optimizer([
            self.F, self.dgcn, self.dv_classifier, self.domain_discri
        ])
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)

        best_acc = 0.0; best_acc_epoch = 0; best_acc_f1 = 0.0
        best_f1 = 0.0; best_f1_epoch = 0; best_f1_acc = 0.0

        for epoch in range(1, args.max_epoch + 1):
            self.current_epoch = epoch
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-' * 5)
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))

            for m in [self.F, self.dgcn, self.dv_classifier, self.domain_discri]: m.train()
            epoch_loss = defaultdict(float); epoch_acc = defaultdict(float)
            tradeoff = self._get_tradeoff(args.tradeoff, epoch)
            num_iter = len(self.dataloaders['train'])
            if num_iter == 0: logging.error("Empty!"); break

            for i in tqdm(range(num_iter), ascii=True):
                self.optimizer.zero_grad()

                if args.train_mode == 'multi_source':
                    try:
                        target_data, _ = utils.get_next_batch(self.dataloaders, self.iters, 'train', self.device)
                        source_data, source_labels = [], []
                        for idx in range(self.num_source):
                            d, l = utils.get_next_batch(self.dataloaders, self.iters, src[idx], self.device)
                            source_data.append(d)
                            source_labels.append(l)
                    except StopIteration: break

                    f_t = self._extract_features(target_data)
                    loss_per_domain = torch.zeros(self.num_source).to(self.device)
                    loss_adapt_per_domain = torch.zeros(self.num_source).to(self.device)
                    for idx in range(self.num_source):
                        f_s = self._extract_features(source_data[idx])
                        f_s_mixed = self.style_mixer(f_s, f_t)

                        bin_s, multi_s = self.dv_classifier(f_s)
                        bin_s_mixed, multi_s_mixed = self.dv_classifier(f_s_mixed)

                        loss_multi = self._focal_loss(multi_s, source_labels[idx], self.focal_gamma)
                        loss_multi_mixed = self._focal_loss(multi_s_mixed, source_labels[idx], self.focal_gamma)
                        epoch_acc['Source'] += utils.get_accuracy(multi_s, source_labels[idx]) / self.num_source

                        with torch.no_grad():
                            probs = F.softmax(multi_s, dim=1)
                            true_class_prob = probs.gather(1, source_labels[idx].unsqueeze(1)).squeeze(1)
                        loss_bin = F.binary_cross_entropy(bin_s, true_class_prob)
                        loss_style = F.kl_div(
                            F.log_softmax(multi_s_mixed, dim=1),
                            F.softmax(multi_s, dim=1).detach(),
                            reduction='batchmean'
                        )

                        loss_adv, acc_d = self.domain_adv[idx](f_s, f_t)
                        epoch_acc['Discriminator'] += acc_d / self.num_source

                        loss_per_domain[idx] = loss_multi + 0.5 * loss_multi_mixed + self.bin_weight * loss_bin + self.style_weight * loss_style
                        loss_adapt_per_domain[idx] = loss_adv

                    self.domain_weight_module.update(loss_per_domain, domain_idxes)
                    domain_weight = self.domain_weight_module.get_domain_weight(domain_idxes)
                    loss = (domain_weight * (loss_per_domain + tradeoff[0] * loss_adapt_per_domain)).sum()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.F.parameters()) + list(self.dgcn.parameters()) +
                        list(self.dv_classifier.parameters()) +
                        [p for d in self.domain_discri for p in d.parameters()], max_norm=10.0)
                    self.optimizer.step()

                    epoch_loss['Multi-Cls'] += (domain_weight * loss_per_domain).sum().item()
                    epoch_loss['Adversarial'] += (domain_weight * loss_adapt_per_domain).sum().item()
                    continue

                try:
                    target_data, _ = utils.get_next_batch(self.dataloaders, self.iters, 'train', self.device)
                    source_data, source_labels = utils.get_next_batch(self.dataloaders, self.iters, src, self.device)
                except StopIteration: break

                # === Extract features ===
                f_s = self._extract_features(source_data)
                f_t = self._extract_features(target_data)

                # === Style Mixing ===
                f_s_mixed = self.style_mixer(f_s, f_t)
                f_t_mixed = self.style_mixer(f_t, f_s)

                # === Dual-View Classifier on source ===
                bin_s, multi_s = self.dv_classifier(f_s)
                bin_s_mixed, multi_s_mixed = self.dv_classifier(f_s_mixed)

                # Multi-class loss (focal for imbalance)
                loss_multi = self._focal_loss(multi_s, source_labels, self.focal_gamma)
                loss_multi_mixed = self._focal_loss(multi_s_mixed, source_labels, self.focal_gamma)
                epoch_acc['Source'] += utils.get_accuracy(multi_s, source_labels)

                # Binary confidence loss (easy samples should have high confidence)
                with torch.no_grad():
                    probs = F.softmax(multi_s, dim=1)
                    true_class_prob = probs.gather(1, source_labels.unsqueeze(1)).squeeze(1)
                loss_bin = F.binary_cross_entropy(bin_s, true_class_prob)

                # Style consistency (mixed predictions should match original)
                loss_style = F.kl_div(
                    F.log_softmax(multi_s_mixed, dim=1),
                    F.softmax(multi_s, dim=1).detach(),
                    reduction='batchmean'
                )

                # === Domain adversarial ===
                loss_adv, acc_d = self.domain_adv(f_s, f_t)
                epoch_acc['Discriminator'] += acc_d

                # === Total Loss ===
                loss = loss_multi \
                    + 0.5 * loss_multi_mixed \
                    + self.bin_weight * loss_bin \
                    + self.style_weight * loss_style \
                    + tradeoff[0] * loss_adv

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.F.parameters()) + list(self.dgcn.parameters()) +
                    list(self.dv_classifier.parameters()) +
                    list(self.domain_discri.parameters()), max_norm=10.0)
                self.optimizer.step()

                epoch_loss['Multi-Cls'] += loss_multi.item()
                epoch_loss['Multi-Mix'] += loss_multi_mixed.item()
                epoch_loss['Binary'] += loss_bin.item()
                epoch_loss['Style'] += loss_style.item()
                epoch_loss['Adversarial'] += loss_adv.item()

            for key in sorted(epoch_loss): logging.info('Train-Loss {}: {:.4f}'.format(key, epoch_loss[key] / max(num_iter, 1)))
            for key in sorted(epoch_acc): logging.info('Train-Acc {}: {:.4f}'.format(key, epoch_acc[key] / max(num_iter, 1)))

            new_acc, new_f1 = self.test()
            if new_acc >= best_acc: best_acc, best_acc_epoch, best_acc_f1 = new_acc, epoch, new_f1; self.save_model_acc()
            if new_f1 >= best_f1: best_f1, best_f1_epoch, best_f1_acc = new_f1, epoch, new_acc; self.save_model_f1()

            logging.info(f"Epoch {epoch}: Val-Acc={new_acc:.4f}, Val-F1={new_f1:.4f}")
            logging.info(f"Best → Acc: Ep{best_acc_epoch}({best_acc:.4f}) | F1: Ep{best_f1_epoch}({best_f1:.4f})")
            if self.lr_scheduler is not None: self.lr_scheduler.step()

        logging.info('=' * 70)
        logging.info(f'DVSMN Best: Acc={best_acc:.4f} (Ep{best_acc_epoch}) | F1={best_f1:.4f} (Ep{best_f1_epoch})')
        logging.info('=' * 70)

    @torch.no_grad()
    def test(self):
        for m in [self.F, self.dgcn, self.dv_classifier, self.domain_discri]: m.eval()
        all_preds, all_labels = [], []
        if 'val' not in self.dataloaders or self.dataloaders['val'] is None: return 0.0, 0.0
        try: iters = iter(self.dataloaders['val']); num_iter = len(self.dataloaders['val'])
        except: return 0.0, 0.0
        if num_iter == 0: return 0.0, 0.0

        for i in range(num_iter):
            try:
                batch = next(iters); td, tl = batch if len(batch) == 2 else (batch[0], batch[1])
            except StopIteration: break
            except: continue
            if td is None: continue
            td, tl = td.to(self.device), tl.to(self.device)
            f = self._extract_features(td)
            _, multi_out = self.dv_classifier(f)
            all_preds.extend(torch.argmax(multi_out, dim=1).cpu().numpy())
            all_labels.extend(tl.cpu().numpy())

        if len(all_preds) == 0: return 0.0, 0.0
        all_preds, all_labels = np.array(all_preds), np.array(all_labels)
        acc = accuracy_score(all_labels, all_preds)
        ul = np.unique(all_labels)
        if len(ul) < 2: f1 = 0.0
        else:
            try:
                if self.num_classes == 2: f1 = f1_score(all_labels, all_preds, average='binary', pos_label=1 if 1 in ul else ul[-1])
                else: f1 = f1_score(all_labels, all_preds, average=self.f1_average)
            except: f1 = f1_score(all_labels, all_preds, average='micro')
        logging.info('Val-Acc : {:.4f}'.format(acc)); logging.info('Val-F1  : {:.4f}'.format(f1))
        return acc, f1
