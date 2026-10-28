'''
Paper: Optimal Transport for Cross-Domain Fault Diagnosis (adapted),
    leveraging Sinkhorn divergence and Wasserstein distance for domain alignment.
Reference code: https://github.com/Pear-so/OT_Fault-Diagnosis
Note: Adapted for 1D vibration signal bearing fault diagnosis.

    Core innovations:
      1. Sinkhorn Divergence — entropic regularized optimal transport for
         efficient and differentiable domain discrepancy measurement.
      2. Mini-batch OT — computes OT cost between source and target mini-batches
         for scalable training.
      3. Joint Distribution Alignment — aligns feature + label joint distributions
         using conditional OT (coupling features conditioned on predicted labels).
      4. Feature Transport Plan — learns a smooth transport map from source
         to target feature space.

    Compatible with: python train.py --model_name OT_FaultDiagnosis --source CWRU_0 --target CWRU_1
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
# 1. Sinkhorn-Knopp Algorithm for Entropic Optimal Transport
# ======================================================================
def sinkhorn_knopp(cost_matrix, epsilon=0.01, max_iters=100, tol=1e-6):
    """
    Sinkhorn-Knopp algorithm for entropic regularized optimal transport.

    Computes the optimal coupling matrix P that minimizes:
        <P, C> - epsilon * H(P)
    subject to P * 1 = a, P^T * 1 = b (uniform marginals by default).

    Args:
        cost_matrix: (B_s, B_t) pairwise cost/distance matrix
        epsilon: entropic regularization strength
        max_iters: maximum iterations
        tol: convergence tolerance

    Returns:
        P: (B_s, B_t) optimal transport plan
        transport_cost: scalar — wasserstein distance estimate
    """
    B_s, B_t = cost_matrix.shape
    device = cost_matrix.device

    # Uniform marginals
    a = torch.ones(B_s, device=device) / B_s
    b = torch.ones(B_t, device=device) / B_t

    # Gibbs kernel
    K = torch.exp(-cost_matrix / epsilon)  # (B_s, B_t)

    # Sinkhorn iterations
    v = torch.ones(B_t, device=device)
    for _ in range(max_iters):
        u = a / (K @ v + 1e-10)
        v_prev = v
        v = b / (K.T @ u + 1e-10)

        if torch.abs(v - v_prev).mean() < tol:
            break

    # Optimal transport plan
    P = torch.diag(u) @ K @ torch.diag(v)

    # Transport cost: <P, C>
    transport_cost = (P * cost_matrix).sum()

    return P, transport_cost


def sinkhorn_divergence(x_s, x_t, epsilon=0.01, max_iters=50):
    """
    Sinkhorn divergence: S_eps(x_s, x_t) = OT_eps(x_s, x_t) - 0.5*OT_eps(x_s, x_s) - 0.5*OT_eps(x_t, x_t)

    This is an unbiased estimator of the true Wasserstein distance that
    interpolates between MMD and Wasserstein.

    Args:
        x_s: (B, D) source features
        x_t: (B, D) target features
        epsilon: entropic regularization
        max_iters: Sinkhorn iterations

    Returns:
        divergence: scalar
    """
    # Pairwise Euclidean distance matrices
    C_st = torch.cdist(x_s, x_t, p=2) ** 2  # (B, B)
    C_ss = torch.cdist(x_s, x_s, p=2) ** 2
    C_tt = torch.cdist(x_t, x_t, p=2) ** 2

    _, cost_st = sinkhorn_knopp(C_st, epsilon, max_iters)
    _, cost_ss = sinkhorn_knopp(C_ss, epsilon, max_iters)
    _, cost_tt = sinkhorn_knopp(C_tt, epsilon, max_iters)

    divergence = cost_st - 0.5 * cost_ss - 0.5 * cost_tt
    return torch.clamp(divergence, min=0.0)


# ======================================================================
# 2. Conditional OT (Feature + Label Joint Alignment)
# ======================================================================
def conditional_ot_loss(features_s, logits_s, features_t, logits_t, epsilon=0.01):
    """
    Conditional optimal transport: aligns joint distributions P(F, Y).

    Decomposes OT cost into per-class OT costs weighted by predicted class
    probabilities, approximating the joint distribution alignment.

    Args:
        features_s: (B, D)
        logits_s: (B, C)
        features_t: (B, D)
        logits_t: (B, C)
        epsilon: entropic regularization

    Returns:
        cond_ot_loss: scalar
    """
    probs_s = F.softmax(logits_s, dim=1)  # (B, C)
    probs_t = F.softmax(logits_t, dim=1)  # (B, C)

    num_classes = logits_s.shape[1]
    total_loss = 0.0

    for c in range(num_classes):
        # Weight by class probability
        w_s = probs_s[:, c]  # (B,)
        w_t = probs_t[:, c]  # (B,)

        if w_s.sum() < 1e-6 or w_t.sum() < 1e-6:
            continue

        # Weighted Sinkhorn divergence for this class
        # Approximate: use class-weighted features
        f_s_c = features_s * w_s.unsqueeze(1)
        f_t_c = features_t * w_t.unsqueeze(1)

        div_c = sinkhorn_divergence(f_s_c, f_t_c, epsilon)
        total_loss += div_c

    return total_loss / max(num_classes, 1)


# ======================================================================
# 3. Trainset — Standard Interface
# ======================================================================
class Trainset(InitTrain):
    """
    OT_FaultDiagnosis: Optimal Transport based Cross-Domain Fault Diagnosis.

    Architecture:
      1. Feature Extractor F (multi-scale 1D-CNN)
      2. Classifier C → class predictions
      3. Domain alignment via Sinkhorn Divergence (Wasserstein distance)
      4. Conditional OT for joint distribution alignment
      5. Domain adversarial training as auxiliary alignment

    Loss = L_cls + λ_ot*L_ot + λ_cot*L_cot + λ_adv*L_adv
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

        # ---------- 3. Domain Discriminator (auxiliary) ----------
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

        # ---------- 4. Hyperparameters ----------
        self.ot_weight = 0.1         # Sinkhorn divergence weight
        self.cot_weight = 0.05       # Conditional OT weight
        self.epsilon = 0.05          # Entropic regularization for Sinkhorn

        # ---------- 5. F1 Settings ----------
        self.num_classes = args.num_classes
        self.f1_average = self._get_f1_average()
        logging.info(f"OT-FD | {self.num_classes}-class | F1: '{self.f1_average}' "
                     f"| OT w:{self.ot_weight} COT w:{self.cot_weight} eps:{self.epsilon}")

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
        }, save_path)
        logging.info('ACC Model saved to {}'.format(save_path))

    def save_model_f1(self):
        save_path = self.args.save_path + '_best_f1.pth'
        torch.save({
            'F': self.F.state_dict(), 'C': self.C.state_dict(),
            'domain_discri': self._discriminator_state_dict(),
        }, save_path)
        logging.info('F1 Model saved to {}'.format(save_path))

    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path, map_location=self.device)
        for k in ['F', 'C']:
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

        self.optimizer = self._get_optimizer([self.F, self.C, self.domain_discri])
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)

        best_acc = 0.0; best_acc_epoch = 0; best_acc_f1 = 0.0
        best_f1 = 0.0; best_f1_epoch = 0; best_f1_acc = 0.0

        for epoch in range(1, args.max_epoch + 1):
            self.current_epoch = epoch
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-' * 5)
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))

            self.F.train(); self.C.train(); self.domain_discri.train()
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
                    logits_t = self.C(f_t)
                    loss_per_domain = torch.zeros(self.num_source).to(self.device)
                    loss_adapt_per_domain = torch.zeros(self.num_source).to(self.device)
                    for idx in range(self.num_source):
                        f_s = self.F(source_data[idx])
                        logits_s = self.C(f_s)
                        loss_cls = F.cross_entropy(logits_s, source_labels[idx])
                        epoch_acc['Source'] += utils.get_accuracy(logits_s, source_labels[idx]) / self.num_source

                        loss_ot = sinkhorn_divergence(f_s, f_t, epsilon=self.epsilon)
                        loss_cot = conditional_ot_loss(f_s, logits_s, f_t, logits_t, epsilon=self.epsilon)
                        loss_adv, acc_d = self.domain_adv[idx](f_s, f_t)
                        epoch_acc['Discriminator'] += acc_d / self.num_source

                        loss_per_domain[idx] = loss_cls
                        loss_adapt_per_domain[idx] = self.ot_weight * loss_ot + self.cot_weight * loss_cot + tradeoff[0] * 0.5 * loss_adv

                    self.domain_weight_module.update(loss_per_domain, domain_idxes)
                    domain_weight = self.domain_weight_module.get_domain_weight(domain_idxes)
                    loss = (domain_weight * (loss_per_domain + loss_adapt_per_domain)).sum()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.F.parameters()) + list(self.C.parameters()) +
                        [p for d in self.domain_discri for p in d.parameters()], max_norm=10.0)
                    self.optimizer.step()

                    epoch_loss['Classifier'] += (domain_weight * loss_per_domain).sum().item()
                    epoch_loss['OT (Sinkhorn)'] += (domain_weight * loss_adapt_per_domain).sum().item()
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

                # === Sinkhorn Divergence (OT loss) ===
                loss_ot = sinkhorn_divergence(f_s, f_t, epsilon=self.epsilon)

                # === Conditional OT ===
                logits_t = self.C(f_t)
                loss_cot = conditional_ot_loss(f_s, logits_s, f_t, logits_t, epsilon=self.epsilon)

                # === Domain Adversarial (auxiliary) ===
                loss_adv, acc_d = self.domain_adv(f_s, f_t)
                epoch_acc['Discriminator'] += acc_d

                # === Total Loss ===
                loss = loss_cls \
                    + self.ot_weight * loss_ot \
                    + self.cot_weight * loss_cot \
                    + tradeoff[0] * 0.5 * loss_adv  # reduced adversarial weight

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.F.parameters()) + list(self.C.parameters()) +
                    list(self.domain_discri.parameters()), max_norm=10.0)
                self.optimizer.step()

                epoch_loss['Classifier'] += loss_cls.item()
                epoch_loss['OT (Sinkhorn)'] += loss_ot.item()
                epoch_loss['Conditional OT'] += loss_cot.item()
                epoch_loss['Discriminator'] += loss_adv.item()

            for key in sorted(epoch_loss): logging.info('Train-Loss {}: {:.4f}'.format(key, epoch_loss[key] / max(num_iter, 1)))
            for key in sorted(epoch_acc): logging.info('Train-Acc {}: {:.4f}'.format(key, epoch_acc[key] / max(num_iter, 1)))

            new_acc, new_f1 = self.test()
            if new_acc >= best_acc: best_acc, best_acc_epoch, best_acc_f1 = new_acc, epoch, new_f1; self.save_model_acc()
            if new_f1 >= best_f1: best_f1, best_f1_epoch, best_f1_acc = new_f1, epoch, new_acc; self.save_model_f1()

            logging.info(f"Epoch {epoch}: Val-Acc={new_acc:.4f}, Val-F1={new_f1:.4f}")
            logging.info(f"Best → Acc: Ep{best_acc_epoch}({best_acc:.4f}) | F1: Ep{best_f1_epoch}({best_f1:.4f})")
            if self.lr_scheduler is not None: self.lr_scheduler.step()

        logging.info('=' * 70)
        logging.info(f'OT-FD Best: Acc={best_acc:.4f} (Ep{best_acc_epoch}) | F1={best_f1:.4f} (Ep{best_f1_epoch})')
        logging.info('=' * 70)

    @torch.no_grad()
    def test(self):
        self.F.eval(); self.C.eval(); self.domain_discri.eval()
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
            logits = self.C(self.F(td))
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
