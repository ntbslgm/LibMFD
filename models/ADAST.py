'''
Paper: Eldele, E., Ragab, M., Chen, Z., Wu, M., Kwoh, C.K., Li, X. and Guan, C., 2023.
    ADAST: Attentive Cross-Domain EEG-Based Sleep Staging Framework with Iterative Self-Training.
    IEEE Transactions on Emerging Topics in Computational Intelligence, 7(1), pp.210-221.
Reference code: https://github.com/emadeldeen24/ADAST
Note: Adapted for 1D vibration signal bearing fault diagnosis.
    Key components:
      1. Shared feature extractor F (multi-scale CNN backbone)
      2. Domain-specific attention modules (psi_s, psi_t) — unshared per-domain feature transformation
      3. Domain adversarial training with gradient reversal layer
      4. Dual classifiers (C1, C2) with weight orthogonality penalty for diversity
      5. Iterative self-training with pseudo-labels from dual-classifier consensus
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
# Domain-Specific Attention Module
# ======================================================================
class DomainAttention(nn.Module):
    """
    Domain-specific bottleneck MLP with residual connection and learnable gate.

    This serves as the per-domain attention transformation (psi_s / psi_t in the paper).
    The unshared parameters allow each domain to preserve its own characteristic features
    while the shared feature extractor F learns domain-invariant representations.

    Args:
        input_dim:  Feature dimension from the shared feature extractor
        bottleneck: Bottleneck dimension (default: input_dim // 10)
        dropout:    Dropout rate
    """
    def __init__(self, input_dim=2560, bottleneck=256, dropout=0.):
        super(DomainAttention, self).__init__()
        self.transform = nn.Sequential(
            nn.Linear(input_dim, bottleneck),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, input_dim),
        )
        # Learnable gate: starts at 0 so attention initially passes through identity
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.gamma * self.transform(x) + x


# ======================================================================
# Trainset — Standard Interface (compatible with train.py)
# ======================================================================
class Trainset(InitTrain):
    """
    ADAST: Attentive Domain Adaptation with Self-Training.

    Combines four innovations for unsupervised domain adaptation:
      1. Shared feature extractor (F) — multi-scale 1D-CNN
      2. Domain-specific attention (psi_s, psi_t) — unshared bottleneck MLPs
      3. Domain adversarial alignment — GRL + discriminator
      4. Dual classifiers (C1, C2) + iterative self-training

    Compatible with the standard train.py interface:
        python train.py --model_name ADAST --source CWRU_0 --target CWRU_1 --train_mode single_source --cuda_device 0
    """

    def __init__(self, args):
        super(Trainset, self).__init__(args)

        output_size = 2560  # FeatureExtractor output dimension

        # ---------- 1. Shared Feature Extractor F ----------
        self.F = model_base.FeatureExtractor(
            in_channel=1, dropout=args.dropout
        ).to(self.device)

        # ---------- 2. Domain-Specific Attention psi_s, psi_t ----------
        self.psi_s = DomainAttention(
            input_dim=output_size, bottleneck=256, dropout=args.dropout
        ).to(self.device)
        self.psi_t = DomainAttention(
            input_dim=output_size, bottleneck=256, dropout=args.dropout
        ).to(self.device)

        # ---------- 3. Dual Classifiers C1, C2 ----------
        self.C1 = nn.Linear(output_size, args.num_classes).to(self.device)
        self.C2 = nn.Linear(output_size, args.num_classes).to(self.device)

        # ---------- Weight Initialization for Stability ----------
        self._init_weights()

        # ---------- 4. Domain Discriminator + Adversarial Loss ----------
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

        # ---------- Loss Function ----------
        self.cross_entropy = nn.CrossEntropyLoss()

        # ---------- Self-Training Hyperparameters ----------
        self.diversity_weight = 0.01   # λ₂: weight for classifier orthogonality penalty
        self.self_training_warmup = 5  # epochs before enabling target self-training at full strength

        # ---------- F1 Settings ----------
        self.num_classes = args.num_classes
        self.f1_average = self._get_f1_average()
        logging.info(f"Task: {self.num_classes}-class classification | F1 average: '{self.f1_average}'")

        # ---------- Load Data ----------
        self._init_data()

    def _get_f1_average(self):
        """Automatically select F1 averaging method based on number of classes."""
        if self.num_classes == 2:
            return 'binary'
        elif self.num_classes > 2:
            return 'weighted'
        else:
            return 'micro'

    def _init_weights(self):
        """Initialize weights with stability-focused schemes.

        DomainAttention: Xavier init for linear layers, small init for the second
        linear to keep residual path dominant early in training (gamma=0 already).
        Classifiers: standard Linear default (kaiming_uniform) is fine.
        """
        for psi in [self.psi_s, self.psi_t]:
            for name, module in psi.transform.named_modules():
                if isinstance(module, nn.Linear):
                    if '0' in name or name == '':  # first linear: standard Xavier
                        nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
                    else:  # second (output) linear: small init
                        nn.init.xavier_uniform_(module.weight, gain=0.01)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

    @staticmethod
    def _validate_tensor(t, name="tensor", repair=False):
        """Check tensor for NaN/Inf. Returns (is_valid, repaired_tensor_or_original)."""
        nan_mask = torch.isnan(t)
        inf_mask = torch.isinf(t)
        if nan_mask.any() or inf_mask.any():
            num_nan = nan_mask.sum().item()
            num_inf = inf_mask.sum().item()
            logging.warning(
                f"[Validate] {name}: {num_nan} NaN, {num_inf} Inf "
                f"(shape={tuple(t.shape)}, range=[{t[~nan_mask & ~inf_mask].min().item():.4f}, "
                f"{t[~nan_mask & ~inf_mask].max().item():.4f}])"
            )
            if repair:
                t = t.clone()
                t[nan_mask | inf_mask] = 0.0
                return False, t
            return False, t
        return True, t

    # ================== Model Save / Load ==================
    def save_model_acc(self):
        save_path = self.args.save_path + '_best_acc.pth'
        torch.save({
            'F': self.F.state_dict(),
            'psi_s': self.psi_s.state_dict(),
            'psi_t': self.psi_t.state_dict(),
            'C1': self.C1.state_dict(),
            'C2': self.C2.state_dict(),
            'domain_discri': self._discriminator_state_dict(),
        }, save_path)
        logging.info('ACC Model saved to {}'.format(save_path))

    def save_model_f1(self):
        save_path = self.args.save_path + '_best_f1.pth'
        torch.save({
            'F': self.F.state_dict(),
            'psi_s': self.psi_s.state_dict(),
            'psi_t': self.psi_t.state_dict(),
            'C1': self.C1.state_dict(),
            'C2': self.C2.state_dict(),
            'domain_discri': self._discriminator_state_dict(),
        }, save_path)
        logging.info('F1 Model saved to {}'.format(save_path))

    def _discriminator_state_dict(self):
        if self.train_mode == 'multi_source':
            return {f'disc_{i}': d.state_dict() for i, d in enumerate(self.domain_discri)}
        return self.domain_discri.state_dict()

    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path, map_location=self.device)
        self.F.load_state_dict(ckpt['F'])
        self.psi_s.load_state_dict(ckpt['psi_s'])
        self.psi_t.load_state_dict(ckpt['psi_t'])
        self.C1.load_state_dict(ckpt['C1'])
        self.C2.load_state_dict(ckpt['C2'])
        if 'domain_discri' in ckpt:
            if self.train_mode == 'multi_source':
                for i, d in enumerate(self.domain_discri):
                    d.load_state_dict(ckpt['domain_discri'][f'disc_{i}'])
            else:
                self.domain_discri.load_state_dict(ckpt['domain_discri'])

    # ================== Training ==================
    def train(self):
        args = self.args

        # Determine source
        if args.train_mode == 'single_source':
            src = args.source_name[0]
        elif args.train_mode == 'source_combine':
            src = args.source_name
        elif args.train_mode == 'multi_source':
            src = args.source_name
            domain_idxes = list(range(self.num_source))

        self.optimizer = self._get_optimizer([
            self.F, self.psi_s, self.psi_t, self.C1, self.C2, self.domain_discri
        ])
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)

        # ---- Gradient clipping threshold (prevents gradient explosion from GRL) ----
        grad_clip = getattr(args, 'grad_clip', 10.0)

        # Initialize best metric tracking
        best_acc = 0.0
        best_acc_epoch = 0
        best_acc_f1 = 0.0

        best_f1 = 0.0
        best_f1_epoch = 0
        best_f1_acc = 0.0

        # ---- NaN tracking for diagnostics ----
        nan_count_epoch = 0
        nan_count_total = 0

        for epoch in range(1, args.max_epoch + 1):
            self.current_epoch = epoch
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-' * 5)

            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))

            # ---- Set train mode ----
            self.F.train()
            self.psi_s.train()
            self.psi_t.train()
            self.C1.train()
            self.C2.train()
            self.domain_discri.train()

            # ---- Self-training weight: λ₁=0 for epoch 1, then ramps up ----
            if epoch <= 1:
                lambda_1 = 0.0
            else:
                lambda_1 = min(1.0, (epoch - 1) / self.self_training_warmup)

            epoch_loss = defaultdict(float)
            epoch_acc = defaultdict(float)
            tradeoff = self._get_tradeoff(args.tradeoff, epoch)
            nan_count_epoch = 0

            num_iter = len(self.dataloaders['train'])
            if num_iter == 0:
                logging.error("Training dataloader is empty!")
                break

            for i in tqdm(range(num_iter), ascii=True):
                # ================================================================
                # Step 0: Fetch & validate input data
                # ================================================================
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

                    self.optimizer.zero_grad()
                    f_t = self.F(target_data)
                    a_t = self.psi_t(f_t)

                    # global diversity penalty (classifier weights, domain-independent)
                    w1_flat = self.C1.weight.view(-1)
                    w2_flat = self.C2.weight.view(-1)
                    loss_div = torch.abs(torch.dot(w1_flat, w2_flat)) / w1_flat.numel()

                    # target self-training loss (domain-independent)
                    loss_tgt = torch.tensor(0.0).to(self.device)
                    if lambda_1 > 0:
                        with torch.no_grad():
                            logits_t1_det = self.C1(a_t)
                            logits_t2_det = self.C2(a_t)
                            probs_t_det = 0.5 * (
                                F.softmax(logits_t1_det, dim=1) +
                                F.softmax(logits_t2_det, dim=1)
                            )
                            pseudo_labels = torch.clamp(
                                torch.argmax(probs_t_det, dim=1), 0, self.num_classes - 1
                            )
                        logits_t1 = self.C1(a_t)
                        logits_t2 = self.C2(a_t)
                        loss_tgt = 0.5 * (
                            self.cross_entropy(logits_t1, pseudo_labels) +
                            self.cross_entropy(logits_t2, pseudo_labels)
                        )
                        if torch.isnan(loss_tgt) or torch.isinf(loss_tgt):
                            loss_tgt = torch.tensor(0.0).to(self.device)

                    loss_per_domain = torch.zeros(self.num_source).to(self.device)
                    loss_adapt_per_domain = torch.zeros(self.num_source).to(self.device)
                    for idx in range(self.num_source):
                        f_s = self.F(source_data[idx])
                        a_s = self.psi_s(f_s)
                        logits_s1 = self.C1(a_s)
                        logits_s2 = self.C2(a_s)
                        loss_cls = self.cross_entropy(logits_s1, source_labels[idx].long()) \
                                 + self.cross_entropy(logits_s2, source_labels[idx].long())
                        epoch_acc['Source C1'] += utils.get_accuracy(logits_s1, source_labels[idx].long()) / self.num_source
                        epoch_acc['Source C2'] += utils.get_accuracy(logits_s2, source_labels[idx].long()) / self.num_source

                        loss_d, acc_d = self.domain_adv[idx](a_s, a_t)
                        epoch_acc['Discriminator'] += acc_d / self.num_source

                        loss_per_domain[idx] = loss_cls
                        loss_adapt_per_domain[idx] = loss_d

                    self.domain_weight_module.update(loss_per_domain, domain_idxes)
                    domain_weight = self.domain_weight_module.get_domain_weight(domain_idxes)
                    loss = (domain_weight * (loss_per_domain + tradeoff[0] * loss_adapt_per_domain)).sum() \
                         + self.diversity_weight * loss_div \
                         + lambda_1 * loss_tgt

                    if torch.isnan(loss) or torch.isinf(loss):
                        logging.warning(f"[Skip] Epoch {epoch} iter {i}: NaN/Inf in total loss — skipping batch")
                        self.optimizer.zero_grad()
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for model in [self.F, self.psi_s, self.psi_t,
                                         self.C1, self.C2] + [d for d in self.domain_discri]
                         for p in model.parameters()],
                        max_norm=grad_clip
                    )
                    self.optimizer.step()

                    epoch_loss['Source Classifier'] += (domain_weight * loss_per_domain).sum().item()
                    epoch_loss['Discriminator'] += (domain_weight * loss_adapt_per_domain).sum().item()
                    epoch_loss['Classifier Diversity'] += loss_div.item()
                    if lambda_1 > 0 and loss_tgt.item() > 0:
                        epoch_loss['Target Self-Training'] += loss_tgt.item()
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

                # ---- Validate input tensors: if data is corrupt, skip this batch ----
                skip_batch = False
                for t, tname in [(source_data, 'source_data'), (target_data, 'target_data')]:
                    ok, _ = self._validate_tensor(t, tname)
                    if not ok:
                        skip_batch = True
                if skip_batch:
                    logging.warning(f"[Skip] Epoch {epoch} iter {i}: input data has NaN/Inf — skipping")
                    continue

                # ---- Validate labels ----
                source_labels = source_labels.long()
                if source_labels.min() < 0 or source_labels.max() >= self.num_classes:
                    logging.warning(
                        f"[Label Error] source_labels range: "
                        f"[{source_labels.min()}, {source_labels.max()}], "
                        f"num_classes={self.num_classes} — clamping!"
                    )
                    source_labels = torch.clamp(source_labels, 0, self.num_classes - 1)

                # ================================================================
                # Step 1: Forward pass with NaN checks at every stage
                # ================================================================
                self.optimizer.zero_grad()

                # ---- 1a. Shared Feature Extractor F ----
                f_s = self.F(source_data)
                f_t = self.F(target_data)
                ok_fs, f_s = self._validate_tensor(f_s, 'f_s', repair=True)
                ok_ft, f_t = self._validate_tensor(f_t, 'f_t', repair=True)
                if not (ok_fs and ok_ft):
                    logging.warning(f"[Skip] Epoch {epoch} iter {i}: NaN in F output — skipping")
                    nan_count_epoch += 1
                    nan_count_total += 1
                    continue

                # ---- 1b. Domain-Specific Attention ----
                a_s = self.psi_s(f_s)
                a_t = self.psi_t(f_t)
                ok_as, a_s = self._validate_tensor(a_s, 'a_s', repair=True)
                ok_at, a_t = self._validate_tensor(a_t, 'a_t', repair=True)
                if not (ok_as and ok_at):
                    logging.warning(f"[Skip] Epoch {epoch} iter {i}: NaN in attention output — skipping")
                    nan_count_epoch += 1
                    nan_count_total += 1
                    continue

                # ---- 1c. Source Classification Logits ----
                logits_s1 = self.C1(a_s)
                logits_s2 = self.C2(a_s)
                ok_s1, _ = self._validate_tensor(logits_s1, 'logits_s1')
                ok_s2, _ = self._validate_tensor(logits_s2, 'logits_s2')
                if not (ok_s1 and ok_s2):
                    logging.warning(f"[Skip] Epoch {epoch} iter {i}: NaN in source logits — skipping")
                    nan_count_epoch += 1
                    nan_count_total += 1
                    continue

                loss_cls = self.cross_entropy(logits_s1, source_labels) \
                         + self.cross_entropy(logits_s2, source_labels)

                epoch_acc['Source C1'] += utils.get_accuracy(logits_s1, source_labels)
                epoch_acc['Source C2'] += utils.get_accuracy(logits_s2, source_labels)

                # ---- 1d. Domain Adversarial Loss ----
                loss_d, acc_d = self.domain_adv(a_s, a_t)
                if torch.isnan(loss_d) or torch.isinf(loss_d):
                    logging.warning(f"[Skip] Epoch {epoch} iter {i}: NaN/Inf in domain loss — skipping")
                    nan_count_epoch += 1
                    nan_count_total += 1
                    continue
                epoch_acc['Discriminator'] += acc_d

                # ---- 1e. Classifier Diversity Penalty ----
                w1_flat = self.C1.weight.view(-1)
                w2_flat = self.C2.weight.view(-1)
                loss_div = torch.abs(torch.dot(w1_flat, w2_flat)) / w1_flat.numel()

                # ---- 1f. Target Self-Training Loss ----
                if lambda_1 > 0:
                    with torch.no_grad():
                        logits_t1_det = self.C1(a_t)
                        logits_t2_det = self.C2(a_t)
                        ok_t1, _ = self._validate_tensor(logits_t1_det, 'logits_t1_det')
                        ok_t2, _ = self._validate_tensor(logits_t2_det, 'logits_t2_det')
                        if not (ok_t1 and ok_t2):
                            # If pseudo-label generation has NaN, skip self-training for this batch only
                            loss_tgt = torch.tensor(0.0).to(self.device)
                            logging.debug(f"[SelfTrain] Epoch {epoch} iter {i}: NaN in target logits — "
                                         "skipping self-training for this batch")
                        else:
                            probs_t_det = 0.5 * (
                                F.softmax(logits_t1_det, dim=1) +
                                F.softmax(logits_t2_det, dim=1)
                            )
                            pseudo_labels = torch.argmax(probs_t_det, dim=1)
                            pseudo_labels = torch.clamp(pseudo_labels, 0, self.num_classes - 1)

                            logits_t1 = self.C1(a_t)
                            logits_t2 = self.C2(a_t)
                            loss_tgt = 0.5 * (
                                self.cross_entropy(logits_t1, pseudo_labels) +
                                self.cross_entropy(logits_t2, pseudo_labels)
                            )
                            # If self-training loss is NaN, drop it gracefully
                            if torch.isnan(loss_tgt) or torch.isinf(loss_tgt):
                                loss_tgt = torch.tensor(0.0).to(self.device)
                                logging.debug(f"[SelfTrain] Epoch {epoch} iter {i}: NaN in self-training loss — "
                                             "skipping self-training for this batch")
                else:
                    loss_tgt = torch.tensor(0.0).to(self.device)

                # ================================================================
                # Step 2: Total loss
                # ================================================================
                loss = loss_cls \
                     + tradeoff[0] * loss_d \
                     + self.diversity_weight * loss_div \
                     + lambda_1 * loss_tgt

                # ---- Final NaN guard on total loss ----
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning(f"[Skip] Epoch {epoch} iter {i}: NaN/Inf in total loss — skipping batch")
                    nan_count_epoch += 1
                    nan_count_total += 1
                    self.optimizer.zero_grad()
                    continue

                # ================================================================
                # Step 3: Backward + Gradient Clipping + Optimizer step
                # ================================================================
                loss.backward()

                # ---- Gradient clipping: prevent explosion from GRL + adversarial ----
                torch.nn.utils.clip_grad_norm_(
                    [p for model in [self.F, self.psi_s, self.psi_t,
                                     self.C1, self.C2, self.domain_discri]
                     for p in model.parameters()],
                    max_norm=grad_clip
                )

                self.optimizer.step()

                # ---- Post-step: check if weights became NaN (indicates unrecoverable state) ----
                for name, param in self.C1.named_parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        logging.error(
                            f"[FATAL] Epoch {epoch} iter {i}: {name} weights became NaN/Inf "
                            f"after optimizer step — training may be unrecoverable!"
                        )

                # ---- Record losses ----
                epoch_loss['Source Classifier'] += loss_cls.item()
                epoch_loss['Discriminator'] += loss_d.item()
                epoch_loss['Classifier Diversity'] += loss_div.item()
                if lambda_1 > 0 and loss_tgt.item() > 0:
                    epoch_loss['Target Self-Training'] += loss_tgt.item()

            # ---- End of epoch ----
            if nan_count_epoch > 0:
                logging.warning(
                    f"[NaN Summary] Epoch {epoch}: {nan_count_epoch}/{num_iter} iterations "
                    f"skipped due to NaN (total: {nan_count_total})"
                )

            # ---- Print training stats ----
            for key in sorted(epoch_loss.keys()):
                logging.info('Train-Loss {}: {:.4f}'.format(
                    key, epoch_loss[key] / max(num_iter, 1)))
            for key in sorted(epoch_acc.keys()):
                logging.info('Train-Acc {}: {:.4f}'.format(
                    key, epoch_acc[key] / max(num_iter, 1)))

            # ---- Validation ----
            new_acc, new_f1 = self.test()

            # ---- Update best Accuracy ----
            if new_acc >= best_acc:
                best_acc = new_acc
                best_acc_epoch = epoch
                best_acc_f1 = new_f1
                self.save_model_acc()

            # ---- Update best F1 ----
            if new_f1 >= best_f1:
                best_f1 = new_f1
                best_f1_epoch = epoch
                best_f1_acc = new_acc
                self.save_model_f1()

            # ---- Log current and best results ----
            logging.info(f"Epoch {epoch}: Val-Acc={new_acc:.4f}, Val-F1={new_f1:.4f}")
            logging.info("Best so far -> "
                         f"Acc: Epoch {best_acc_epoch} (Acc={best_acc:.4f}, F1={best_acc_f1:.4f}) | "
                         f"F1: Epoch {best_f1_epoch} (Acc={best_f1_acc:.4f}, F1={best_f1:.4f})")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # ---- Training Summary ----
        logging.info('=' * 70)
        logging.info('Training finished! Best results summary:')
        logging.info(f'Best Accuracy: {best_acc:.4f} (Epoch {best_acc_epoch}, F1={best_acc_f1:.4f})')
        logging.info(f'Best F1-score: {best_f1:.4f} (Epoch {best_f1_epoch}, Acc={best_f1_acc:.4f})')
        if nan_count_total > 0:
            logging.warning(f'Total NaN iterations skipped: {nan_count_total}')
        logging.info('=' * 70)

    # ================== Validation ==================
    @torch.no_grad()
    def test(self):
        self.F.eval()
        self.psi_s.eval()
        self.psi_t.eval()
        self.C1.eval()
        self.C2.eval()
        self.domain_discri.eval()

        all_preds = []
        all_labels = []

        # Check if validation set exists
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

            # Forward: F → psi_t → C1+C2 (averaged)
            f_t = self.F(target_data)
            a_t = self.psi_t(f_t)
            logits = 0.5 * (self.C1(a_t) + self.C2(a_t))

            pred_labels = torch.argmax(logits, dim=1)

            all_preds.extend(pred_labels.cpu().numpy())
            all_labels.extend(target_labels.cpu().numpy())

        # Check for valid predictions
        if len(all_preds) == 0:
            logging.warning("No predictions made during validation!")
            return 0.0, 0.0

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Calculate accuracy
        acc = accuracy_score(all_labels, all_preds)

        # Handle edge case: only one class present
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
