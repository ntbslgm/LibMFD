'''
Paper: Discrete Wavelet Convolutional Network with Cross-Contrast Perturbation
    for Single Domain Generalization in Fault Diagnosis, MSSP 2025.
Reference code: https://github.com/JC952/DWCN-SDGFD
Note: Adapted for 1D vibration signal bearing fault diagnosis.

    Core innovations:
      1. Learnable Discrete Wavelet Operators — parameterized wavelet filters
         for noise-robust multi-scale feature extraction.
      2. Cross-Contrast Perturbation — between main and auxiliary networks,
         perturbing features to enhance diversity while preserving semantics.
      3. Instance Similarity Learning — contrastive-style learning on feature
         space characterization for improved single-domain generalization.

    Compatible with: python train.py --model_name DWCN_SDGFD --source CWRU_0 --target CWRU_1
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
# 1. Learnable Discrete Wavelet Transform (DWT) Layer
# ======================================================================
class LearnableDWT(nn.Module):
    """
    Learnable Discrete Wavelet Transform: parameterized low-pass and high-pass
    filters that decompose the signal into approximation and detail coefficients.

    Args:
        in_channels:  Input channels (1 for 1D vibration)
        levels:       Number of wavelet decomposition levels
        kernel_size:  Length of the wavelet filter kernel
    """
    def __init__(self, in_channels=1, levels=3, kernel_size=8):
        super(LearnableDWT, self).__init__()
        self.levels = levels
        self.kernel_size = kernel_size

        # Learnable low-pass and high-pass filter banks for each level
        self.lp_filters = nn.ModuleList([
            nn.Conv1d(in_channels if i == 0 else in_channels,
                      in_channels,
                      kernel_size=kernel_size,
                      stride=2,
                      padding=kernel_size // 2,
                      bias=False)
            for i in range(levels)
        ])
        self.hp_filters = nn.ModuleList([
            nn.Conv1d(in_channels if i == 0 else in_channels,
                      in_channels,
                      kernel_size=kernel_size,
                      stride=2,
                      padding=kernel_size // 2,
                      bias=False)
            for i in range(levels)
        ])

        # Initialize filters to approximate Daubechies wavelet
        self._init_wavelet_filters()

    def _init_wavelet_filters(self):
        """Initialize with approximate wavelet-like coefficients."""
        for lp, hp in zip(self.lp_filters, self.hp_filters):
            # Low-pass: smooth averaging
            nn.init.dirac_(lp.weight)
            # High-pass: difference operator
            with torch.no_grad():
                hp.weight.data = lp.weight.data.clone()
                hp.weight.data[:, :, ::2] *= -1

    def forward(self, x):
        """
        Args:
            x: (B, C, L) input signal
        Returns:
            coeffs: list of [(B, C, L/2^i), (B, C, L/2^i)] for each level
        """
        coeffs = []
        for level in range(self.levels):
            approx = self.lp_filters[level](x)     # Low-pass (approximation)
            detail = self.hp_filters[level](x)     # High-pass (detail)
            coeffs.append((approx, detail))
            x = approx  # Cascade: next level operates on approximation
        return coeffs


class WaveletFeatureExtractor(nn.Module):
    """
    Extract features from wavelet coefficients using parallel 1D-CNNs.
    Each decomposition level gets its own small CNN, then features are fused.
    """
    def __init__(self, in_channels=1, levels=3, hidden_dim=64, output_dim=256, dropout=0.):
        super(WaveletFeatureExtractor, self).__init__()
        self.levels = levels

        # CNN for each level's approximation + detail coefficients
        self.approx_cnns = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, hidden_dim, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(4),
                nn.Flatten(),
            ) for _ in range(levels)
        ])
        self.detail_cnns = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, hidden_dim, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(4),
                nn.Flatten(),
            ) for _ in range(levels)
        ])

        # Fusion layer
        total_dim = levels * 2 * hidden_dim * 4  # levels * (approx + detail) * hidden_dim * pool_size
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, output_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
        )

    def forward(self, coeffs):
        features = []
        for level in range(self.levels):
            approx, detail = coeffs[level]
            f_a = self.approx_cnns[level](approx)
            f_d = self.detail_cnns[level](detail)
            features.extend([f_a, f_d])
        fused = self.fusion(torch.cat(features, dim=1))
        return fused


# ======================================================================
# 2. Cross-Contrast Perturbation Module
# ======================================================================
class CrossContrastPerturbation(nn.Module):
    """
    Cross-Contrast Perturbation between main and auxiliary networks.

    The auxiliary network receives perturbed (noise-injected) features while
    the main network receives original features. The auxiliary network provides
    a regularization signal via prediction consistency.

    Perturbation types:
      - Gaussian noise injection
      - Dropout-based masking
      - Feature shuffling across batch
    """
    def __init__(self, noise_std=0.05, drop_prob=0.1):
        super(CrossContrastPerturbation, self).__init__()
        self.noise_std = noise_std
        self.drop_prob = drop_prob

    def perturb(self, features):
        """
        Apply multiple perturbation strategies.
        Args:
            features: (B, D)
        Returns:
            features_perturbed: (B, D)
        """
        # 1. Gaussian noise
        noise = torch.randn_like(features) * self.noise_std * features.std(dim=0, keepdim=True)
        f_pert = features + noise

        # 2. Random dropout masking
        mask = (torch.rand_like(features) > self.drop_prob).float()
        f_pert = f_pert * mask

        return f_pert


# ======================================================================
# 3. Instance Similarity Learning
# ======================================================================
def instance_similarity_loss(features, temperature=0.07):
    """
    Instance-level similarity learning: pull features of the same class
    closer and push features of different classes apart.

    This uses an unsupervised contrastive approach on the feature space:
    each sample is similar to its augmented version (same sample, different
    perturbation). Without labels, we use the feature space self-similarity.

    Args:
        features: (B, D) — features from main branch
    Returns:
        sim_loss: scalar
    """
    B = features.size(0)
    device = features.device

    # L2 normalize
    f_norm = F.normalize(features, dim=1)

    # Cosine similarity matrix
    sim_matrix = torch.matmul(f_norm, f_norm.T) / temperature  # (B, B)

    # Mask self-similarity
    mask = torch.eye(B, device=device).bool()
    sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

    # Encourage uniform distribution on the hypersphere:
    # penalize high similarity between different samples
    sim_loss = torch.logsumexp(sim_matrix, dim=1).mean()

    return sim_loss


# ======================================================================
# 4. Auxiliary Classifier (lightweight, for contrastive perturbation)
# ======================================================================
class AuxClassifier(nn.Module):
    def __init__(self, input_dim=256, num_classes=10, dropout=0.):
        super(AuxClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ======================================================================
# 5. Trainset — Standard Interface
# ======================================================================
class Trainset(InitTrain):
    """
    DWCN-SDGFD: Discrete Wavelet Convolutional Network with Cross-Contrast
    Perturbation for Single Domain Generalization Fault Diagnosis.

    Architecture:
      1. Shared Feature Extractor F (multi-scale 1D-CNN)
      2. Learnable DWT → Wavelet features
      3. Cross-Contrast Perturbation → main + auxiliary branches
      4. Main Classifier C_main + Auxiliary Classifier C_aux
      5. Domain adversarial alignment (for UDA extension)
      6. Instance Similarity Learning + Perturbation Consistency

    Loss = L_cls_main + 0.5*L_cls_aux + λ_adv*L_adv + λ_sim*L_sim + λ_ccp*L_ccp
    """

    def __init__(self, args):
        super(Trainset, self).__init__(args)

        base_dim = 2560
        wavelet_dim = 256

        # ---------- 1. Time-domain Feature Extractor F ----------
        self.F = model_base.FeatureExtractor(
            in_channel=1, dropout=args.dropout
        ).to(self.device)

        # ---------- 2. Learnable DWT + Wavelet Feature Extractor ----------
        self.dwt = LearnableDWT(in_channels=1, levels=3, kernel_size=8).to(self.device)
        self.wavelet_feat = WaveletFeatureExtractor(
            in_channels=1, levels=3, hidden_dim=64, output_dim=wavelet_dim, dropout=args.dropout
        ).to(self.device)

        # ---------- 3. Feature Fusion ----------
        fused_dim = base_dim + wavelet_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(args.dropout),
            nn.Linear(512, 256),
        ).to(self.device)

        # ---------- 4. Main Classifier ----------
        self.C_main = model_base.ClassifierMLP(
            input_size=256, output_size=args.num_classes,
            dropout=args.dropout, last=None
        ).to(self.device)

        # ---------- 5. Auxiliary Classifier (for perturbed features) ----------
        self.C_aux = AuxClassifier(
            input_dim=256, num_classes=args.num_classes, dropout=args.dropout
        ).to(self.device)

        # ---------- 6. Cross-Contrast Perturbation ----------
        self.ccp = CrossContrastPerturbation(noise_std=0.05, drop_prob=0.1)

        # ---------- 7. Domain Discriminator ----------
        self.train_mode = args.train_mode
        grl = utils.GradientReverseLayer()
        if self.train_mode == 'multi_source':
            self.domain_discri = nn.ModuleList([
                model_base.ClassifierMLP(
                    input_size=256, output_size=1, dropout=args.dropout, last='sigmoid'
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
                input_size=256, output_size=1, dropout=args.dropout, last='sigmoid'
            ).to(self.device)
            self.domain_adv = utils.DomainAdversarialLoss(self.domain_discri, grl=grl)
            self.domain_weight_module = None

        # ---------- 8. Hyperparameters ----------
        self.sim_weight = 0.1       # Instance similarity loss weight
        self.ccp_weight = 0.2       # Cross-contrast perturbation consistency weight
        self.temperature = 0.07     # Similarity temperature

        # ---------- 9. F1 Settings ----------
        self.num_classes = args.num_classes
        self.f1_average = self._get_f1_average()
        logging.info(f"DWCN-SDGFD | {self.num_classes}-class | F1: '{self.f1_average}'")

        self._init_data()

    def _get_f1_average(self):
        if self.num_classes == 2: return 'binary'
        elif self.num_classes > 2: return 'weighted'
        else: return 'micro'

    def _extract_fused(self, x):
        """Extract time-domain, wavelet, and fused features for input x."""
        f_time = self.F(x)                                      # (B, 2560)
        wave_coeffs = self.dwt(x)                               # wavelet decomp
        f_wavelet = self.wavelet_feat(wave_coeffs)              # (B, 256)
        f_fused = self.fusion(torch.cat([f_time, f_wavelet], dim=1))  # (B, 256)
        return f_fused

    def _discriminator_state_dict(self):
        if self.train_mode == 'multi_source':
            return {f'disc_{i}': d.state_dict() for i, d in enumerate(self.domain_discri)}
        return self.domain_discri.state_dict()

    def save_model_acc(self):
        save_path = self.args.save_path + '_best_acc.pth'
        torch.save({
            'F': self.F.state_dict(), 'dwt': self.dwt.state_dict(),
            'wavelet_feat': self.wavelet_feat.state_dict(),
            'fusion': self.fusion.state_dict(), 'C_main': self.C_main.state_dict(),
            'C_aux': self.C_aux.state_dict(), 'domain_discri': self._discriminator_state_dict(),
        }, save_path)
        logging.info('ACC Model saved to {}'.format(save_path))

    def save_model_f1(self):
        save_path = self.args.save_path + '_best_f1.pth'
        torch.save({
            'F': self.F.state_dict(), 'dwt': self.dwt.state_dict(),
            'wavelet_feat': self.wavelet_feat.state_dict(),
            'fusion': self.fusion.state_dict(), 'C_main': self.C_main.state_dict(),
            'C_aux': self.C_aux.state_dict(), 'domain_discri': self._discriminator_state_dict(),
        }, save_path)
        logging.info('F1 Model saved to {}'.format(save_path))

    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path, map_location=self.device)
        for k in ['F', 'dwt', 'wavelet_feat', 'fusion', 'C_main', 'C_aux']:
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
            self.F, self.dwt, self.wavelet_feat, self.fusion,
            self.C_main, self.C_aux, self.domain_discri
        ])
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)

        best_acc = 0.0; best_acc_epoch = 0; best_acc_f1 = 0.0
        best_f1 = 0.0; best_f1_epoch = 0; best_f1_acc = 0.0

        for epoch in range(1, args.max_epoch + 1):
            self.current_epoch = epoch
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-' * 5)
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))

            for m in [self.F, self.dwt, self.wavelet_feat, self.fusion,
                      self.C_main, self.C_aux, self.domain_discri]:
                m.train()

            epoch_loss = defaultdict(float); epoch_acc = defaultdict(float)
            tradeoff = self._get_tradeoff(args.tradeoff, epoch)
            num_iter = len(self.dataloaders['train'])
            if num_iter == 0: logging.error("Empty train loader!"); break

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

                    f_t = self._extract_fused(target_data)
                    loss_per_domain = torch.zeros(self.num_source).to(self.device)
                    loss_adapt_per_domain = torch.zeros(self.num_source).to(self.device)
                    for idx in range(self.num_source):
                        f_s = self._extract_fused(source_data[idx])
                        logits_s_main = self.C_main(f_s)
                        loss_cls_main = F.cross_entropy(logits_s_main, source_labels[idx])
                        epoch_acc['Main'] += utils.get_accuracy(logits_s_main, source_labels[idx]) / self.num_source

                        f_s_pert = self.ccp.perturb(f_s)
                        logits_s_aux = self.C_aux(f_s_pert)
                        loss_cls_aux = F.cross_entropy(logits_s_aux, source_labels[idx])
                        epoch_acc['Aux'] += utils.get_accuracy(logits_s_aux, source_labels[idx]) / self.num_source

                        loss_ccp = F.kl_div(
                            F.log_softmax(logits_s_aux, dim=1),
                            F.softmax(logits_s_main, dim=1).detach(),
                            reduction='batchmean'
                        )

                        loss_adv, acc_d = self.domain_adv[idx](f_s, f_t)
                        epoch_acc['Discriminator'] += acc_d / self.num_source
                        loss_sim = instance_similarity_loss(f_s, self.temperature)

                        loss_per_domain[idx] = loss_cls_main + 0.5 * loss_cls_aux
                        loss_adapt_per_domain[idx] = loss_adv + self.sim_weight * loss_sim + self.ccp_weight * loss_ccp

                    self.domain_weight_module.update(loss_per_domain, domain_idxes)
                    domain_weight = self.domain_weight_module.get_domain_weight(domain_idxes)
                    loss = (domain_weight * (loss_per_domain + tradeoff[0] * loss_adapt_per_domain)).sum()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.F.parameters()) + list(self.dwt.parameters()) +
                        list(self.wavelet_feat.parameters()) + list(self.fusion.parameters()) +
                        list(self.C_main.parameters()) + list(self.C_aux.parameters()) +
                        [p for d in self.domain_discri for p in d.parameters()], max_norm=10.0)
                    self.optimizer.step()

                    epoch_loss['Main Cls'] += (domain_weight * loss_per_domain).sum().item()
                    epoch_loss['Discriminator'] += (domain_weight * loss_adapt_per_domain).sum().item()
                    continue

                try:
                    target_data, _ = utils.get_next_batch(self.dataloaders, self.iters, 'train', self.device)
                    source_data, source_labels = utils.get_next_batch(self.dataloaders, self.iters, src, self.device)
                except StopIteration: break

                # Extract fused features
                f_s = self._extract_fused(source_data)  # (B, 256)
                f_t = self._extract_fused(target_data)  # (B, 256)

                # === Main Classifier ===
                logits_s_main = self.C_main(f_s)
                loss_cls_main = F.cross_entropy(logits_s_main, source_labels)
                epoch_acc['Main'] += utils.get_accuracy(logits_s_main, source_labels)

                # === Cross-Contrast Perturbation ===
                f_s_pert = self.ccp.perturb(f_s)
                logits_s_aux = self.C_aux(f_s_pert)
                loss_cls_aux = F.cross_entropy(logits_s_aux, source_labels)
                epoch_acc['Aux'] += utils.get_accuracy(logits_s_aux, source_labels)

                # Perturbation consistency: main and aux should agree
                loss_ccp = F.kl_div(
                    F.log_softmax(logits_s_aux, dim=1),
                    F.softmax(logits_s_main, dim=1).detach(),
                    reduction='batchmean'
                )

                # === Domain Adversarial ===
                loss_adv, acc_d = self.domain_adv(f_s, f_t)
                epoch_acc['Discriminator'] += acc_d

                # === Instance Similarity (on source) ===
                loss_sim = instance_similarity_loss(f_s, self.temperature)

                # === Total Loss ===
                loss = loss_cls_main \
                    + 0.5 * loss_cls_aux \
                    + tradeoff[0] * loss_adv \
                    + self.sim_weight * loss_sim \
                    + self.ccp_weight * loss_ccp

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.F.parameters()) + list(self.dwt.parameters()) +
                    list(self.wavelet_feat.parameters()) + list(self.fusion.parameters()) +
                    list(self.C_main.parameters()) + list(self.C_aux.parameters()) +
                    list(self.domain_discri.parameters()), max_norm=10.0)
                self.optimizer.step()

                epoch_loss['Main Cls'] += loss_cls_main.item()
                epoch_loss['Aux Cls'] += loss_cls_aux.item()
                epoch_loss['Discriminator'] += loss_adv.item()
                epoch_loss['CCP'] += loss_ccp.item()
                epoch_loss['Sim'] += loss_sim.item()

            for key in sorted(epoch_loss): logging.info('Train-Loss {}: {:.4f}'.format(key, epoch_loss[key] / max(num_iter, 1)))
            for key in sorted(epoch_acc): logging.info('Train-Acc {}: {:.4f}'.format(key, epoch_acc[key] / max(num_iter, 1)))

            new_acc, new_f1 = self.test()
            if new_acc >= best_acc: best_acc, best_acc_epoch, best_acc_f1 = new_acc, epoch, new_f1; self.save_model_acc()
            if new_f1 >= best_f1: best_f1, best_f1_epoch, best_f1_acc = new_f1, epoch, new_acc; self.save_model_f1()

            logging.info(f"Epoch {epoch}: Val-Acc={new_acc:.4f}, Val-F1={new_f1:.4f}")
            logging.info(f"Best → Acc: Ep{best_acc_epoch}({best_acc:.4f},F1={best_acc_f1:.4f}) | "
                         f"F1: Ep{best_f1_epoch}({best_f1_acc:.4f},F1={best_f1:.4f})")
            if self.lr_scheduler is not None: self.lr_scheduler.step()

        logging.info('=' * 70)
        logging.info(f'DWCN-SDGFD Best: Acc={best_acc:.4f} (Ep{best_acc_epoch}) | F1={best_f1:.4f} (Ep{best_f1_epoch})')
        logging.info('=' * 70)

    def test(self):
        for m in [self.F, self.dwt, self.wavelet_feat, self.fusion,
                  self.C_main, self.C_aux, self.domain_discri]: m.eval()

        all_preds, all_labels = [], []

        # check if validation dataloader exists
        if 'val' not in self.dataloaders or self.dataloaders['val'] is None:
            logging.warning("No validation dataloader found!")
            return 0.0, 0.0

        try:
            iters = iter(self.dataloaders['val'])
            num_iter = len(self.dataloaders['val'])
        except Exception as e:
            logging.warning(f"Failed to create validation iterator: {e}")
            return 0.0, 0.0

        if num_iter == 0:
            logging.warning("Validation dataloader is empty!")
            return 0.0, 0.0

        with torch.no_grad():
            for i in range(num_iter):
                try:
                    batch = next(iters)
                    if len(batch) == 2:
                        td, tl = batch
                    else:
                        td, tl, _ = batch
                except StopIteration:
                    break
                except Exception as e:
                    logging.warning(f"Error getting validation batch: {e}")
                    continue

                if td is None or tl is None:
                    continue

                td, tl = td.to(self.device), tl.to(self.device)
                f = self._extract_fused(td)
                logits = self.C_main(f)

                # handle case where model returns tuple
                if isinstance(logits, tuple):
                    logits = logits[0]

                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_labels.extend(tl.cpu().numpy())

        if len(all_preds) == 0:
            logging.warning("No predictions made during validation!")
            return 0.0, 0.0

        all_preds, all_labels = np.array(all_preds), np.array(all_labels)
        acc = accuracy_score(all_labels, all_preds)

        ul = np.unique(all_labels)
        if len(ul) < 2:
            f1 = 0.0
            logging.warning(f"Only {len(ul)} class(es) present in validation set. F1 score is not defined.")
        else:
            try:
                if self.num_classes == 2:
                    pos_label = 1 if 1 in ul else ul[-1]
                    f1 = f1_score(all_labels, all_preds, average='binary', pos_label=pos_label)
                else:
                    f1 = f1_score(all_labels, all_preds, average=self.f1_average)
            except Exception as e:
                logging.warning(f"Error calculating F1: {e}")
                f1 = f1_score(all_labels, all_preds, average='micro')

        logging.info('Val-Acc : {:.4f}'.format(acc))
        logging.info('Val-F1  : {:.4f}'.format(f1))
        return acc, f1
