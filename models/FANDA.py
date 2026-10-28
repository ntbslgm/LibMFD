'''
Paper: Chai, Z. and Zhao, C., 2019. A Fine-Grained Adversarial Network
    Method for Cross-Domain Industrial Fault Diagnosis. IEEE TASE.
Reference code: https://github.com/chunhuiz/FANDA-for-Cross-Domain-Industrial-Fault-Diagnosis
Note: Adapted for 1D vibration signal bearing fault diagnosis.

    Core innovations:
      1. Feature Generator — extracts fault features from raw signals.
      2. Fault Classifier — recognizes fault types from extracted features.
      3. Fine-Grained Domain Discriminators — per-class domain discriminators
         that align each fault class distribution across domains, preserving
         the fault-discriminative structure.
      4. Global Domain Discriminator — reduces overall distribution discrepancy
         between source and target domains.
      5. The key insight: traditional domain adversarial methods only align
         global distributions, but FANDA performs BOTH global and fine-grained
         (per-class) alignment.

    Compatible with: python train.py --model_name FANDA --source CWRU_0 --target CWRU_1
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
# 1. Fine-Grained Domain Discriminator
# ======================================================================
class FineGrainedDiscriminator(nn.Module):
    """
    Per-class domain discriminator: aligns source and target distributions
    separately for each fault class.

    Input: [feature, class_probability_weight]
    Architecture: uses class probabilities as attention weights to focus
    the discriminator on samples belonging to that specific class.
    """
    def __init__(self, input_dim, num_classes, hidden_dim=512, dropout=0.):
        super(FineGrainedDiscriminator, self).__init__()

        # Per-class discriminators (shared trunk + separate heads)
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Per-class output heads
        self.class_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            ) for _ in range(num_classes)
        ])

        self.num_classes = num_classes
        self.current_accuracy = 0.0

    def forward(self, features, class_probs):
        """
        Args:
            features: (B, D) feature vectors
            class_probs: (B, C) class probability distribution (softmax)
        Returns:
            d_outputs: (B, C) per-class domain predictions
        """
        h = self.shared(features)  # (B, H)
        # Weight features by class probability for each class
        d_list = []
        for c in range(self.num_classes):
            d_c = self.class_heads[c](h)  # (B, 1)
            d_list.append(d_c)
        d_outputs = torch.cat(d_list, dim=1)  # (B, C)
        return d_outputs


class FineGrainedAdversarialLoss(nn.Module):
    """
    Fine-grained adversarial loss: aligns per-class distributions.

    For each class c, the loss is:
        L_c = BCE(D_c(f_s), 1) + BCE(D_c(f_t), 0)
    weighted by the class probability p(y=c|x).

    This ensures that alignment happens primarily for the classes
    that each sample actually belongs to.
    """
    def __init__(self, discriminator, num_classes, grl=None):
        super(FineGrainedAdversarialLoss, self).__init__()
        self.grl = utils.WarmStartGradientReverseLayer(
            alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True
        ) if grl is None else grl
        self.discriminator = discriminator
        self.num_classes = num_classes
        self.accuracy = 0.0

    def forward(self, f_s, prob_s, f_t, prob_t):
        """
        Args:
            f_s, f_t: (B, D) source/target features
            prob_s, prob_t: (B, C) softmax class probabilities
        Returns:
            loss: scalar — weighted fine-grained adversarial loss
            acc: scalar — discriminator accuracy
        """
        # Apply GRL
        f_all = self.grl(torch.cat([f_s, f_t], dim=0))
        prob_all = torch.cat([prob_s, prob_t], dim=0)

        d_out = self.discriminator(f_all, prob_all)  # (2B, C)
        d_s, d_t = d_out.chunk(2, dim=0)  # (B, C), (B, C)

        # Domain labels
        d_label_s = torch.ones_like(d_s)
        d_label_t = torch.zeros_like(d_t)

        # Class-probability weighted BCE
        # Higher weight for classes the sample likely belongs to
        loss_per_class_s = F.binary_cross_entropy(d_s, d_label_s, reduction='none')
        loss_per_class_t = F.binary_cross_entropy(d_t, d_label_t, reduction='none')

        # Weight by class probabilities
        weighted_loss_s = (loss_per_class_s * prob_s).sum(dim=1).mean()
        weighted_loss_t = (loss_per_class_t * prob_t).sum(dim=1).mean()

        loss = 0.5 * (weighted_loss_s + weighted_loss_t)

        # Accuracy
        d_pred = (d_out > 0.5).float()
        d_true = torch.cat([d_label_s, d_label_t], dim=0)
        self.accuracy = (d_pred == d_true).float().mean()

        return loss


# ======================================================================
# 2. Feature Generator (reuses model_base) + Classifier
# ======================================================================
# FANDA uses a unified BaseModel (FeatureExtractor + Classifier)
# and adds fine-grained + global discriminators


# ======================================================================
# 3. Trainset — Standard Interface
# ======================================================================
class Trainset(InitTrain):
    """
    FANDA: Fine-Grained Adversarial Network for Domain Adaptation.

    Architecture:
      1. Feature Generator G (FeatureExtractor) → base features
      2. Fault Classifier C → class predictions (also provides class probs
         for fine-grained alignment)
      3. Global Domain Discriminator D_global → overall distribution alignment
      4. Fine-Grained Domain Discriminators D_fg → per-class alignment

    Loss = L_cls + λ_global*L_global + λ_fg*L_fg

    Key distinction: D_fg uses class probabilities to weight domain
    discrimination per class, ensuring each fault type's distribution
    is aligned separately across domains.
    """

    def __init__(self, args):
        super(Trainset, self).__init__(args)

        output_size = 2560

        # ---------- 1. Feature Generator ----------
        self.G = model_base.FeatureExtractor(
            in_channel=1, dropout=args.dropout
        ).to(self.device)

        # ---------- 2. Fault Classifier ----------
        self.C = model_base.ClassifierMLP(
            input_size=output_size, output_size=args.num_classes,
            dropout=args.dropout, last=None
        ).to(self.device)

        # ---------- 3. Global Domain Discriminator ----------
        self.train_mode = args.train_mode
        grl_global = utils.GradientReverseLayer()
        if self.train_mode == 'multi_source':
            self.D_global = nn.ModuleList([
                model_base.ClassifierMLP(
                    input_size=output_size, output_size=1,
                    dropout=args.dropout, last='sigmoid'
                ) for _ in range(self.num_source)
            ]).to(self.device)
            self.global_adv = nn.ModuleList([
                utils.DomainAdversarialLoss(d, grl=grl_global) for d in self.D_global
            ])

            # ---------- 4. Fine-Grained Domain Discriminator ----------
            self.D_fg = nn.ModuleList([
                FineGrainedDiscriminator(
                    input_dim=output_size, num_classes=args.num_classes,
                    hidden_dim=512, dropout=args.dropout
                ) for _ in range(self.num_source)
            ]).to(self.device)
            self.fg_adv = nn.ModuleList([
                FineGrainedAdversarialLoss(d, num_classes=args.num_classes) for d in self.D_fg
            ])
            self.domain_weight_module = utils.AutomaticUpdateDomainWeightModule(
                num_domains=self.num_source, eta=1e-2, device=self.device
            )
        else:
            self.D_global = model_base.ClassifierMLP(
                input_size=output_size, output_size=1,
                dropout=args.dropout, last='sigmoid'
            ).to(self.device)
            self.global_adv = utils.DomainAdversarialLoss(self.D_global, grl=grl_global)

            # ---------- 4. Fine-Grained Domain Discriminator ----------
            self.D_fg = FineGrainedDiscriminator(
                input_dim=output_size, num_classes=args.num_classes,
                hidden_dim=512, dropout=args.dropout
            ).to(self.device)
            self.fg_adv = FineGrainedAdversarialLoss(
                self.D_fg, num_classes=args.num_classes
            )
            self.domain_weight_module = None

        # ---------- 5. Hyperparameters ----------
        self.fg_weight = 0.5  # Fine-grained adversarial weight

        # ---------- 6. F1 Settings ----------
        self.num_classes = args.num_classes
        self.f1_average = self._get_f1_average()
        logging.info(f"FANDA | {self.num_classes}-class | F1: '{self.f1_average}'")
        logging.info(f"FANDA | FG weight: {self.fg_weight}")

        self._init_data()

    def _get_f1_average(self):
        if self.num_classes == 2: return 'binary'
        elif self.num_classes > 2: return 'weighted'
        else: return 'micro'

    def _save_disc(self, name, module):
        if self.train_mode == 'multi_source':
            return {f'disc_{i}': d.state_dict() for i, d in enumerate(module)}
        return module.state_dict()

    def _load_disc(self, name, module, state):
        if self.train_mode == 'multi_source':
            for i, d in enumerate(module):
                d.load_state_dict(state[f'disc_{i}'])
        else:
            module.load_state_dict(state)

    def save_model_acc(self):
        save_path = self.args.save_path + '_best_acc.pth'
        torch.save({
            'G': self.G.state_dict(), 'C': self.C.state_dict(),
            'D_global': self._save_disc('D_global', self.D_global),
            'D_fg': self._save_disc('D_fg', self.D_fg),
        }, save_path)
        logging.info('ACC Model saved to {}'.format(save_path))

    def save_model_f1(self):
        save_path = self.args.save_path + '_best_f1.pth'
        torch.save({
            'G': self.G.state_dict(), 'C': self.C.state_dict(),
            'D_global': self._save_disc('D_global', self.D_global),
            'D_fg': self._save_disc('D_fg', self.D_fg),
        }, save_path)
        logging.info('F1 Model saved to {}'.format(save_path))

    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path, map_location=self.device)
        for k in ['G', 'C']:
            if k in ckpt: getattr(self, k).load_state_dict(ckpt[k])
        if 'D_global' in ckpt: self._load_disc('D_global', self.D_global, ckpt['D_global'])
        if 'D_fg' in ckpt: self._load_disc('D_fg', self.D_fg, ckpt['D_fg'])

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
            self.G, self.C, self.D_global, self.D_fg
        ])
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)

        best_acc = 0.0; best_acc_epoch = 0; best_acc_f1 = 0.0
        best_f1 = 0.0; best_f1_epoch = 0; best_f1_acc = 0.0

        for epoch in range(1, args.max_epoch + 1):
            self.current_epoch = epoch
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-' * 5)
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))

            for m in [self.G, self.C, self.D_global, self.D_fg]: m.train()
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

                    f_t = self.G(target_data)
                    logits_t = self.C(f_t)
                    prob_t = F.softmax(logits_t, dim=1).detach()

                    loss_per_domain = torch.zeros(self.num_source).to(self.device)
                    loss_adapt_per_domain = torch.zeros(self.num_source).to(self.device)
                    for idx in range(self.num_source):
                        f_s = self.G(source_data[idx])
                        logits_s = self.C(f_s)
                        loss_cls = F.cross_entropy(logits_s, source_labels[idx])
                        epoch_acc['Source'] += utils.get_accuracy(logits_s, source_labels[idx]) / self.num_source

                        prob_s = F.softmax(logits_s, dim=1).detach()

                        loss_global, acc_global = self.global_adv[idx](f_s, f_t)
                        epoch_acc['D_global'] += acc_global / self.num_source

                        loss_fg = self.fg_adv[idx](f_s, prob_s, f_t, prob_t)
                        epoch_acc['D_fg'] += self.fg_adv[idx].accuracy / self.num_source

                        loss_per_domain[idx] = loss_cls
                        loss_adapt_per_domain[idx] = loss_global + self.fg_weight * loss_fg

                    self.domain_weight_module.update(loss_per_domain, domain_idxes)
                    domain_weight = self.domain_weight_module.get_domain_weight(domain_idxes)
                    loss = (domain_weight * (loss_per_domain + tradeoff[0] * loss_adapt_per_domain)).sum()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.G.parameters()) + list(self.C.parameters()) +
                        [p for d in self.D_global for p in d.parameters()] +
                        [p for d in self.D_fg for p in d.parameters()], max_norm=10.0)
                    self.optimizer.step()

                    epoch_loss['Classifier'] += (domain_weight * loss_per_domain).sum().item()
                    epoch_loss['Adv'] += (domain_weight * loss_adapt_per_domain).sum().item()
                    continue

                try:
                    target_data, _ = utils.get_next_batch(self.dataloaders, self.iters, 'train', self.device)
                    source_data, source_labels = utils.get_next_batch(self.dataloaders, self.iters, src, self.device)
                except StopIteration: break

                # === Feature extraction ===
                f_s = self.G(source_data)
                f_t = self.G(target_data)

                # === Classification ===
                logits_s = self.C(f_s)
                loss_cls = F.cross_entropy(logits_s, source_labels)
                epoch_acc['Source'] += utils.get_accuracy(logits_s, source_labels)

                # === Softmax probabilities (for fine-grained alignment) ===
                prob_s = F.softmax(logits_s, dim=1).detach()  # Detach to avoid gradient issues
                logits_t = self.C(f_t)
                prob_t = F.softmax(logits_t, dim=1).detach()

                # === Global Domain Adversarial ===
                loss_global, acc_global = self.global_adv(f_s, f_t)
                epoch_acc['D_global'] += acc_global

                # === Fine-Grained Domain Adversarial ===
                loss_fg = self.fg_adv(f_s, prob_s, f_t, prob_t)
                epoch_acc['D_fg'] += self.fg_adv.accuracy

                # === Total Loss ===
                loss = loss_cls \
                    + tradeoff[0] * loss_global \
                    + self.fg_weight * loss_fg

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.G.parameters()) + list(self.C.parameters()) +
                    list(self.D_global.parameters()) + list(self.D_fg.parameters()),
                    max_norm=10.0)
                self.optimizer.step()

                epoch_loss['Classifier'] += loss_cls.item()
                epoch_loss['Global Adv'] += loss_global.item()
                epoch_loss['Fine-Grained Adv'] += loss_fg.item()

            for key in sorted(epoch_loss): logging.info('Train-Loss {}: {:.4f}'.format(key, epoch_loss[key] / max(num_iter, 1)))
            for key in sorted(epoch_acc): logging.info('Train-Acc {}: {:.4f}'.format(key, epoch_acc[key] / max(num_iter, 1)))

            new_acc, new_f1 = self.test()
            if new_acc >= best_acc: best_acc, best_acc_epoch, best_acc_f1 = new_acc, epoch, new_f1; self.save_model_acc()
            if new_f1 >= best_f1: best_f1, best_f1_epoch, best_f1_acc = new_f1, epoch, new_acc; self.save_model_f1()

            logging.info(f"Epoch {epoch}: Val-Acc={new_acc:.4f}, Val-F1={new_f1:.4f}")
            logging.info(f"Best → Acc: Ep{best_acc_epoch}({best_acc:.4f}) | F1: Ep{best_f1_epoch}({best_f1:.4f})")
            if self.lr_scheduler is not None: self.lr_scheduler.step()

        logging.info('=' * 70)
        logging.info(f'FANDA Best: Acc={best_acc:.4f} (Ep{best_acc_epoch}) | F1={best_f1:.4f} (Ep{best_f1_epoch})')
        logging.info('=' * 70)

    @torch.no_grad()
    def test(self):
        for m in [self.G, self.C, self.D_global, self.D_fg]: m.eval()
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
            logits = self.C(self.G(td))
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
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
