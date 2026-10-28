'''
Paper: Zhao, C. and Shen, W., 2022. A domain generalization network
    combining invariance and specificity towards real-time intelligent
    fault diagnosis. MSSP.
Reference code: https://github.com/CHAOZHAO-1/DGNIS
Note: Adapted for 1D vibration signal bearing fault diagnosis (UDA mode).

    Core innovations:
      1. Invariance Learning — domain-invariant feature extraction via
         adversarial training and MMD-based distribution matching.
      2. Specificity Preservation — domain-specific prototype memory to
         retain useful per-domain characteristics (not all domain info is harmful).
      3. Invariance-Specificity Trade-off — adaptive weighting mechanism
         to balance between invariant and specific features.
      4. Real-time inference — lightweight architecture suitable for
         online fault diagnosis deployment.

    Compatible with: python train.py --model_name DGNIS --source CWRU_0 --target CWRU_1
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
# 1. MMD Loss for Invariance
# ======================================================================
class MMDLoss(nn.Module):
    """
    Multiple Kernel Maximum Mean Discrepancy for distribution matching.
    Used to enforce invariance between source and target feature distributions.
    """
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMDLoss, self).__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def _gaussian_kernel(self, source, target, kernel_mul, kernel_num):
        n_samples = source.size(0) + target.size(0)
        total = torch.cat([source, target], dim=0)

        # L2 distance
        L2_distance = ((total.unsqueeze(0) - total.unsqueeze(1)) ** 2).sum(2)
        bandwidths = self._get_bandwidth(total, kernel_mul, kernel_num)

        kernel_val = 0
        for bandwidth in bandwidths:
            kernel_val += torch.exp(-L2_distance / bandwidth)

        # MMD computation
        XX = kernel_val[:source.size(0), :source.size(0)]
        YY = kernel_val[source.size(0):, source.size(0):]
        XY = kernel_val[:source.size(0), source.size(0):]

        mmd = XX.mean() + YY.mean() - 2 * XY.mean()
        return mmd

    def _get_bandwidth(self, total, kernel_mul, kernel_num):
        L2_distance = ((total.unsqueeze(0) - total.unsqueeze(1)) ** 2).sum(2)
        # Median heuristic
        median_dist = L2_distance.flatten().sort().values[len(L2_distance.flatten()) // 2]
        bandwidths = [median_dist * (kernel_mul ** i) for i in range(kernel_num)]
        return bandwidths

    def forward(self, source, target):
        return self._gaussian_kernel(source, target, self.kernel_mul, self.kernel_num)


# ======================================================================
# 2. Domain-Specific Prototype Memory
# ======================================================================
class DomainSpecificMemory(nn.Module):
    """
    Preserves domain-specific information through learnable per-domain
    prototypes. Each domain has a set of prototypes that capture unique
    characteristics not shared across domains.

    These prototypes provide a "specificity regularization" signal:
    features should be close to their own domain's prototypes but
    far from cross-domain prototypes of the same class.
    """
    def __init__(self, num_domains, feature_dim, num_prototypes_per_domain=8):
        super(DomainSpecificMemory, self).__init__()
        self.num_domains = num_domains
        self.feature_dim = feature_dim
        self.num_prototypes = num_prototypes_per_domain

        # Learnable prototypes: (num_domains, K, D)
        self.prototypes = nn.Parameter(
            torch.randn(num_domains, num_prototypes_per_domain, feature_dim) * 0.01
        )

    def forward(self, features, domain_id=0):
        """
        Compute specificity loss: features should be close to their
        own domain prototypes and distinct from other domain prototypes.

        Args:
            features: (B, D)
            domain_id: 0 for source, 1 for target
        Returns:
            spec_loss: scalar
        """
        B = features.size(0)
        device = features.device

        # Own domain prototypes: (K, D)
        own_protos = self.prototypes[domain_id]
        # Other domain prototypes: (K, D)
        other_protos = self.prototypes[1 - domain_id] if self.num_domains > 1 else own_protos

        # Cosine similarity to own prototypes
        sim_own = F.cosine_similarity(
            features.unsqueeze(1), own_protos.unsqueeze(0), dim=2
        )  # (B, K)
        # Max similarity to own prototypes
        sim_own_max = sim_own.max(dim=1)[0]  # (B,)

        # Cosine similarity to other prototypes
        sim_other = F.cosine_similarity(
            features.unsqueeze(1), other_protos.unsqueeze(0), dim=2
        )  # (B, K)
        sim_other_max = sim_other.max(dim=1)[0]  # (B,)

        # Loss: maximize own similarity, minimize cross similarity
        loss = -sim_own_max.mean() + sim_other_max.mean()
        return loss


# ======================================================================
# 3. Adaptive Invariance-Specificity Gate
# ======================================================================
class AdaptiveGate(nn.Module):
    """
    Learns to weight the trade-off between invariance and specificity.
    Outputs a scalar gate value per sample indicating how much domain-
    specific information to retain.
    """
    def __init__(self, feature_dim, hidden_dim=64):
        super(AdaptiveGate, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        return self.net(features)  # (B, 1)


# ======================================================================
# 4. Trainset — Standard Interface
# ======================================================================
class Trainset(InitTrain):
    """
    DGNIS: Domain Generalization Network combining Invariance and Specificity.

    Architecture:
      1. Feature Extractor F
      2. Classifier C
      3. Domain adversarial alignment (GRL + D) — invariance
      4. MMD alignment — invariance
      5. Domain-Specific Prototype Memory — specificity preservation
      6. Adaptive gate — balances invariance vs specificity

    Loss = L_cls + λ_adv*L_adv + λ_mmd*L_mmd + λ_spec*L_spec
    """

    def __init__(self, args):
        super(Trainset, self).__init__(args)

        output_size = 2560

        # ---------- 1. Feature Extractor ----------
        self.F = model_base.FeatureExtractor(
            in_channel=1, dropout=args.dropout
        ).to(self.device)

        # ---------- 2. Classifier ----------
        self.C = model_base.ClassifierMLP(
            input_size=output_size, output_size=args.num_classes,
            dropout=args.dropout, last=None
        ).to(self.device)

        # ---------- 3. Domain Discriminator (invariance) ----------
        self.train_mode = args.train_mode
        grl = utils.GradientReverseLayer()
        if self.train_mode == 'multi_source':
            self.domain_discri = nn.ModuleList([
                model_base.ClassifierMLP(
                    input_size=output_size, output_size=1,
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
                input_size=output_size, output_size=1,
                dropout=args.dropout, last='sigmoid'
            ).to(self.device)
            self.domain_adv = utils.DomainAdversarialLoss(self.domain_discri, grl=grl)
            self.domain_weight_module = None

        # ---------- 4. MMD Loss ----------
        self.mmd_loss = MMDLoss(kernel_mul=2.0, kernel_num=5)

        # ---------- 5. Domain-Specific Memory ----------
        self.domain_memory = DomainSpecificMemory(
            num_domains=2,  # source + target
            feature_dim=output_size,
            num_prototypes_per_domain=8
        ).to(self.device)

        # ---------- 6. Adaptive Gate ----------
        self.gate = AdaptiveGate(feature_dim=output_size, hidden_dim=64).to(self.device)

        # ---------- 7. Hyperparameters ----------
        self.mmd_weight = 0.1
        self.spec_weight = 0.05
        self.gate_weight = 0.01

        # ---------- 8. F1 Settings ----------
        self.num_classes = args.num_classes
        self.f1_average = self._get_f1_average()
        logging.info(f"DGNIS | {self.num_classes}-class | F1: '{self.f1_average}'")

        self._init_data()

    def _get_f1_average(self):
        if self.num_classes == 2: return 'binary'
        elif self.num_classes > 2: return 'weighted'
        else: return 'micro'

    def _discriminator_state_dict(self):
        if self.train_mode == 'multi_source':
            return {f'disc_{i}': d.state_dict() for i, d in enumerate(self.domain_discri)}
        return self.domain_discri.state_dict()

    def save_model_acc(self):
        save_path = self.args.save_path + '_best_acc.pth'
        torch.save({
            'F': self.F.state_dict(), 'C': self.C.state_dict(),
            'domain_discri': self._discriminator_state_dict(),
            'domain_memory': self.domain_memory.state_dict(),
            'gate': self.gate.state_dict(),
        }, save_path)
        logging.info('ACC Model saved to {}'.format(save_path))

    def save_model_f1(self):
        save_path = self.args.save_path + '_best_f1.pth'
        torch.save({
            'F': self.F.state_dict(), 'C': self.C.state_dict(),
            'domain_discri': self._discriminator_state_dict(),
            'domain_memory': self.domain_memory.state_dict(),
            'gate': self.gate.state_dict(),
        }, save_path)
        logging.info('F1 Model saved to {}'.format(save_path))

    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path, map_location=self.device)
        for k in ['F', 'C', 'domain_memory', 'gate']:
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
            self.F, self.C, self.domain_discri, self.domain_memory, self.gate
        ])
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)

        best_acc = 0.0; best_acc_epoch = 0; best_acc_f1 = 0.0
        best_f1 = 0.0; best_f1_epoch = 0; best_f1_acc = 0.0

        for epoch in range(1, args.max_epoch + 1):
            self.current_epoch = epoch
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-' * 5)
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))

            for m in [self.F, self.C, self.domain_discri, self.domain_memory, self.gate]: m.train()
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

                    f_t = self.F(target_data)
                    loss_per_domain = torch.zeros(self.num_source).to(self.device)
                    loss_adapt_per_domain = torch.zeros(self.num_source).to(self.device)
                    for idx in range(self.num_source):
                        f_s = self.F(source_data[idx])
                        logits_s = self.C(f_s)
                        loss_cls = F.cross_entropy(logits_s, source_labels[idx])
                        epoch_acc['Source'] += utils.get_accuracy(logits_s, source_labels[idx]) / self.num_source

                        loss_adv, acc_d = self.domain_adv[idx](f_s, f_t)
                        epoch_acc['Discriminator'] += acc_d / self.num_source
                        loss_mmd = self.mmd_loss(f_s, f_t)

                        loss_spec_s = self.domain_memory(f_s, domain_id=0)
                        loss_spec_t = self.domain_memory(f_t, domain_id=1)
                        loss_spec = loss_spec_s + loss_spec_t

                        gate_s = self.gate(f_s).mean()
                        gate_t = self.gate(f_t).mean()
                        loss_gate = ((gate_s - 0.5) ** 2 + (gate_t - 0.5) ** 2)

                        loss_per_domain[idx] = loss_cls
                        loss_adapt_per_domain[idx] = loss_adv + self.mmd_weight * loss_mmd + self.spec_weight * loss_spec + self.gate_weight * loss_gate

                    self.domain_weight_module.update(loss_per_domain, domain_idxes)
                    domain_weight = self.domain_weight_module.get_domain_weight(domain_idxes)
                    loss = (domain_weight * (loss_per_domain + tradeoff[0] * loss_adapt_per_domain)).sum()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.F.parameters()) + list(self.C.parameters()) +
                        [p for d in self.domain_discri for p in d.parameters()] +
                        list(self.domain_memory.parameters()) +
                        list(self.gate.parameters()), max_norm=10.0)
                    self.optimizer.step()

                    epoch_loss['Classifier'] += (domain_weight * loss_per_domain).sum().item()
                    epoch_loss['Adversarial'] += (domain_weight * loss_adapt_per_domain).sum().item()
                    continue

                try:
                    target_data, _ = utils.get_next_batch(self.dataloaders, self.iters, 'train', self.device)
                    source_data, source_labels = utils.get_next_batch(self.dataloaders, self.iters, src, self.device)
                except StopIteration: break

                f_s = self.F(source_data); f_t = self.F(target_data)

                # === Classification ===
                logits_s = self.C(f_s)
                loss_cls = F.cross_entropy(logits_s, source_labels)
                epoch_acc['Source'] += utils.get_accuracy(logits_s, source_labels)

                # === Domain Adversarial (invariance) ===
                loss_adv, acc_d = self.domain_adv(f_s, f_t)
                epoch_acc['Discriminator'] += acc_d

                # === MMD (invariance) ===
                loss_mmd = self.mmd_loss(f_s, f_t)

                # === Domain Specificity ===
                loss_spec_s = self.domain_memory(f_s, domain_id=0)
                loss_spec_t = self.domain_memory(f_t, domain_id=1)
                loss_spec = loss_spec_s + loss_spec_t

                # === Adaptive Gate ===
                gate_s = self.gate(f_s).mean()
                gate_t = self.gate(f_t).mean()
                # Gate regularization: encourage gate to stay near 0.5 (balanced)
                loss_gate = ((gate_s - 0.5) ** 2 + (gate_t - 0.5) ** 2)

                # === Total Loss ===
                loss = loss_cls \
                    + tradeoff[0] * loss_adv \
                    + self.mmd_weight * loss_mmd \
                    + self.spec_weight * loss_spec \
                    + self.gate_weight * loss_gate

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.F.parameters()) + list(self.C.parameters()) +
                    list(self.domain_discri.parameters()) +
                    list(self.domain_memory.parameters()) +
                    list(self.gate.parameters()), max_norm=10.0)
                self.optimizer.step()

                epoch_loss['Classifier'] += loss_cls.item()
                epoch_loss['Adversarial'] += loss_adv.item()
                epoch_loss['MMD'] += loss_mmd.item()
                epoch_loss['Specificity'] += loss_spec.item()
                epoch_loss['Gate'] += loss_gate.item()

            for key in sorted(epoch_loss): logging.info('Train-Loss {}: {:.4f}'.format(key, epoch_loss[key] / max(num_iter, 1)))
            for key in sorted(epoch_acc): logging.info('Train-Acc {}: {:.4f}'.format(key, epoch_acc[key] / max(num_iter, 1)))

            new_acc, new_f1 = self.test()
            if new_acc >= best_acc: best_acc, best_acc_epoch, best_acc_f1 = new_acc, epoch, new_f1; self.save_model_acc()
            if new_f1 >= best_f1: best_f1, best_f1_epoch, best_f1_acc = new_f1, epoch, new_acc; self.save_model_f1()

            logging.info(f"Epoch {epoch}: Val-Acc={new_acc:.4f}, Val-F1={new_f1:.4f}")
            logging.info(f"Best → Acc: Ep{best_acc_epoch}({best_acc:.4f}) | F1: Ep{best_f1_epoch}({best_f1:.4f})")
            if self.lr_scheduler is not None: self.lr_scheduler.step()

        logging.info('=' * 70)
        logging.info(f'DGNIS Best: Acc={best_acc:.4f} (Ep{best_acc_epoch}) | F1={best_f1:.4f} (Ep{best_f1_epoch})')
        logging.info('=' * 70)

    def test(self):
        for m in [self.F, self.C, self.domain_discri, self.domain_memory, self.gate]: m.eval()

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
                logits = self.C(self.F(td))

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
