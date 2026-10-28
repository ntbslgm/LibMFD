'''
Paper: Ragab, M., Eldele, E., Chen, Z., Wu, M., Kwoh, C.K., Li, X., 2022.
    Self-supervised Autoregressive Domain Adaptation for Time Series Data (SLARDA).
    IEEE Transactions on Neural Networks and Learning Systems (TNNLS).
Reference code: https://github.com/mohamedr002/SLARDA
Note: Adapted for 1D vibration signal bearing fault diagnosis.
    Three key innovations:
      1. Autoregressive domain discriminator — GRU on sequential CNN features
         captures temporal dependencies during domain alignment
      2. Mean Teacher — EMA-updated teacher model generates stable pseudo-labels
      3. Confident pseudo-labeling — only high-confidence target predictions
         participate in self-training loss
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import logging
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from copy import deepcopy

import utils
import model_base
from train_utils import InitTrain


# ======================================================================
# 1. Split CNN Layer — returns both vector and sequential features
# ======================================================================
class SplitCNNlayer(nn.Module):
    """
    Same architecture as model_base.CNNlayer but returns both:
      - vec_feat: (B, 512) — after AdaptiveMaxPool1d(4) + Flatten
      - seq_feat: (B, 128, L) — before AdaptiveMaxPool (preserves temporal dim)
        where L varies by kernel_size (e.g. ks=4→62, ks=8→54, ks=32→30).
        Later unified via AdaptiveAvgPool1d in SLARDAFeatureExtractor.

    Architecture (identical to model_base.CNNlayer):
      layer1: Conv1d(1→4) → BN → ReLU → MaxPool(2,2)  → (B, 4, 512)
      layer2: Conv1d(4→16) → BN → ReLU → MaxPool(2,2) → (B, 16, 256)
      layer3: Conv1d(16→32) → BN → ReLU → MaxPool(2,2) → (B, 32, 128)
      layer4: Conv1d(32→64) → BN → ReLU → MaxPool(2,2) → (B, 64, 64)
      dropout
      layer5_conv: Conv1d(64→128)→BN→ReLU → (B, 128, 64)  ← seq_feat
      layer5_pool: AdaptiveMaxPool1d(4)→Flatten → (B, 512) ← vec_feat
    """
    def __init__(self, in_channel=1, kernel_size=8, stride=1, padding=1,
                 mp_kernel_size=2, mp_stride=2, dropout=0.):
        super(SplitCNNlayer, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 4, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=mp_kernel_size, stride=mp_stride))

        self.layer2 = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=mp_kernel_size, stride=mp_stride))

        self.layer3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=mp_kernel_size, stride=mp_stride))

        self.layer4 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=mp_kernel_size, stride=mp_stride))

        self.dropout = nn.Dropout(dropout)

        # Split layer5: conv body vs pool tail
        self.layer5_conv = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True))

        self.layer5_pool = nn.Sequential(
            nn.AdaptiveMaxPool1d(4),
            nn.Flatten())

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.dropout(h)
        seq = self.layer5_conv(h)       # (B, 128, 64)
        vec = self.layer5_pool(seq)     # (B, 512)
        return vec, seq


# ======================================================================
# 2. SLARDA Feature Extractor — multi-scale CNN with sequential output
# ======================================================================
class SLARDAFeatureExtractor(nn.Module):
    """
    Multi-scale 1D-CNN feature extractor that returns both:
      - vec_feat: (B, 2560) — concatenated pooled features (for classification)
      - seq_feat: (B, 640, T) — concatenated sequential features (for AR discriminator)

    5 parallel SplitCNNlayer branches with kernel sizes [4, 8, 16, 24, 32].
    Different kernel sizes produce different temporal lengths, so we unify
    via AdaptiveAvgPool1d(16). Total channels: 5 × 128 = 640 (seq),
    5 × 512 = 2560 (vec).
    """
    def __init__(self, in_channel=1, window_sizes=None, dropout=0., seq_pool=16):
        super(SLARDAFeatureExtractor, self).__init__()
        if window_sizes is None:
            window_sizes = [4, 8, 16, 24, 32]
        self.seq_pool = seq_pool
        self.branches = nn.ModuleList([
            SplitCNNlayer(in_channel=in_channel, kernel_size=h, dropout=dropout)
            for h in window_sizes
        ])

    def forward(self, x):
        vec_list, seq_list = [], []
        for branch in self.branches:
            v, s = branch(x)
            vec_list.append(v)
            # Unify temporal dim across branches with different kernel sizes
            s = F.adaptive_avg_pool1d(s, self.seq_pool)
            seq_list.append(s)
        vec = torch.cat(vec_list, dim=1)   # (B, 2560)
        seq = torch.cat(seq_list, dim=1)   # (B, 640, seq_pool)
        return vec, seq


# ======================================================================
# 3. Sequential Domain Discriminator (Autoregressive)
# ======================================================================
class SeqDiscriminator(nn.Module):
    """
    Autoregressive domain discriminator:
      1. 1x1 Conv projects channels: 640 → 128
      2. GRU processes temporal sequence (16 timesteps after AdaptiveAvgPool)
      3. FC layer outputs domain logit

    Input:  (B, 640, T) — T=16 unified sequential CNN features
    Output: (B, 1)      — domain logit (before sigmoid)
    """
    def __init__(self, in_channels=640, proj_channels=128, gru_hidden=64,
                 gru_layers=2, bidirectional=True):
        super(SeqDiscriminator, self).__init__()
        self.channel_proj = nn.Conv1d(in_channels, proj_channels, kernel_size=1)
        self.gru = nn.GRU(
            proj_channels, gru_hidden,
            num_layers=gru_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        gru_out_dim = gru_hidden * (2 if bidirectional else 1)
        self.fc = nn.Linear(gru_out_dim, 1)

    def forward(self, x):
        # x: (B, 640, 64) → Conv1d(1x1) → (B, 128, 64)
        x = self.channel_proj(x)
        # → (B, 64, 128) for GRU (batch_first)
        x = x.permute(0, 2, 1)
        # GRU → output (B, 64, gru_hidden*2), h_n (layers*2, B, gru_hidden)
        output, _ = self.gru(x)
        # Take last timestep
        last = output[:, -1, :]  # (B, gru_hidden*2)
        return self.fc(last)


# ======================================================================
# 4. Trainset — Standard Interface (compatible with train.py)
# ======================================================================
class Trainset(InitTrain):
    """
    SLARDA: Self-supervised Autoregressive Domain Adaptation.

    Three core components:
      1. Shared feature extractor F (multi-scale CNN with sequential output)
      2. Autoregressive domain discriminator D (GRU on sequential features)
      3. Mean Teacher + confident pseudo-labeling for target self-training

    Compatible with the standard train.py interface:
        python train.py --model_name SLARDA --source CWRU_0 --target CWRU_1 --train_mode single_source --cuda_device 0
    """

    def __init__(self, args):
        super(Trainset, self).__init__(args)

        output_size = 2560  # Vec feature dimension

        # ---------- 1. Shared Feature Extractor F ----------
        self.F = SLARDAFeatureExtractor(
            in_channel=1,
            window_sizes=[4, 8, 16, 24, 32],
            dropout=args.dropout
        ).to(self.device)

        # ---------- 2. Classifiers ----------
        self.C1 = nn.Linear(output_size, args.num_classes).to(self.device)  # source cls
        self.C2 = nn.Linear(output_size, args.num_classes).to(self.device)  # target cls

        # ---------- 3. Autoregressive Domain Discriminator ----------
        self.discriminator = SeqDiscriminator(
            in_channels=640, proj_channels=128, gru_hidden=64,
            gru_layers=2, bidirectional=True
        ).to(self.device)

        # ---------- 4. Mean Teacher (EMA copy of F + C2) ----------
        self.teacher_F = SLARDAFeatureExtractor(
            in_channel=1, window_sizes=[4, 8, 16, 24, 32], dropout=args.dropout
        ).to(self.device)
        self.teacher_C = nn.Linear(output_size, args.num_classes).to(self.device)

        # Freeze teacher params (never get gradients)
        for p in self.teacher_F.parameters():
            p.requires_grad = False
        for p in self.teacher_C.parameters():
            p.requires_grad = False

        # ---------- Loss Functions ----------
        self.cross_entropy = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        # ---------- Hyperparameters ----------
        self.teacher_weight = 0.5       # λ: weight for target self-training loss
        self.confidence_threshold = 0.9  # Minimum confidence for pseudo-label
        self.ema_alpha = 0.99           # EMA momentum for teacher update

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

        # ---------- Initialize teacher weights from F + C2 ----------
        self.teacher_F.load_state_dict(self.F.state_dict())
        self.teacher_C.load_state_dict(self.C2.state_dict())

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
            'F': self.F.state_dict(),
            'C1': self.C1.state_dict(),
            'C2': self.C2.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'teacher_F': self.teacher_F.state_dict(),
            'teacher_C': self.teacher_C.state_dict(),
        }, save_path)
        logging.info('ACC Model saved to {}'.format(save_path))

    def save_model_f1(self):
        save_path = self.args.save_path + '_best_f1.pth'
        torch.save({
            'F': self.F.state_dict(),
            'C1': self.C1.state_dict(),
            'C2': self.C2.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'teacher_F': self.teacher_F.state_dict(),
            'teacher_C': self.teacher_C.state_dict(),
        }, save_path)
        logging.info('F1 Model saved to {}'.format(save_path))

    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path, map_location=self.device)
        self.F.load_state_dict(ckpt['F'])
        self.C1.load_state_dict(ckpt['C1'])
        self.C2.load_state_dict(ckpt['C2'])
        if 'discriminator' in ckpt:
            self.discriminator.load_state_dict(ckpt['discriminator'])
        if 'teacher_F' in ckpt:
            self.teacher_F.load_state_dict(ckpt['teacher_F'])
        if 'teacher_C' in ckpt:
            self.teacher_C.load_state_dict(ckpt['teacher_C'])

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

        # Optimizers
        self.optimizer_enc = self._get_optimizer([self.F, self.C1, self.C2])
        self.optimizer_disc = self._get_optimizer([self.discriminator])
        self.lr_scheduler_enc = self._get_lr_scheduler(self.optimizer_enc)
        self.lr_scheduler_disc = self._get_lr_scheduler(self.optimizer_disc)

        # Best metric tracking
        best_acc = 0.0
        best_acc_epoch = 0
        best_acc_f1 = 0.0

        best_f1 = 0.0
        best_f1_epoch = 0
        best_f1_acc = 0.0

        for epoch in range(1, args.max_epoch + 1):
            self.current_epoch = epoch
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-' * 5)

            if self.lr_scheduler_enc is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler_enc.get_last_lr()))

            # ---- Set train mode ----
            self.F.train()
            self.C1.train()
            self.C2.train()
            self.discriminator.train()
            self.teacher_F.eval()
            self.teacher_C.eval()

            # ---- Metrics tracking ----
            epoch_loss = defaultdict(float)
            epoch_acc = defaultdict(float)

            num_iter = len(self.dataloaders['train'])
            if num_iter == 0:
                logging.error("Training dataloader is empty!")
                break

            for i in tqdm(range(num_iter), ascii=True):
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

                    vec_t, seq_t = self.F(target_data)

                    # --- Discriminator step (concatenate all sources vs target) ---
                    self.optimizer_disc.zero_grad()
                    self.optimizer_enc.zero_grad()
                    all_seq_s = [self.F(d)[1].detach() for d in source_data]
                    seq_concat = torch.cat(all_seq_s + [seq_t.detach()], dim=0)
                    disc_pred = self.discriminator(seq_concat).squeeze()
                    n_src_total = sum(s.size(0) for s in all_seq_s)
                    label_concat = torch.cat([
                        torch.ones(n_src_total, device=self.device),
                        torch.zeros(seq_t.size(0), device=self.device)
                    ], dim=0)
                    loss_disc = self.bce_loss(disc_pred, label_concat)
                    loss_disc.backward()
                    self.optimizer_disc.step()
                    disc_acc = ((torch.sigmoid(disc_pred) > 0.5).float() == label_concat).float().mean()
                    epoch_acc['Discriminator'] += disc_acc.item()

                    # --- Encoder step (per-domain weighted source cls + global adv/self-training) ---
                    self.optimizer_disc.zero_grad()
                    self.optimizer_enc.zero_grad()
                    loss_per_domain = torch.zeros(self.num_source).to(self.device)
                    for idx in range(self.num_source):
                        vec_s, seq_s = self.F(source_data[idx])
                        logits_s = self.C1(vec_s)
                        loss_per_domain[idx] = self.cross_entropy(logits_s, source_labels[idx])
                        epoch_acc['Source C1'] += utils.get_accuracy(logits_s, source_labels[idx]) / self.num_source

                    self.domain_weight_module.update(loss_per_domain, domain_idxes)
                    domain_weight = self.domain_weight_module.get_domain_weight(domain_idxes)
                    loss_cls = (domain_weight * loss_per_domain).sum()

                    vec_t, seq_t = self.F(target_data)
                    disc_pred_t = self.discriminator(seq_t).squeeze()
                    label_adv = torch.ones(seq_t.size(0), device=self.device)
                    loss_adv = self.bce_loss(disc_pred_t, label_adv)

                    with torch.no_grad():
                        tea_vec_t, _ = self.teacher_F(target_data)
                        tea_logits = self.teacher_C(tea_vec_t)
                        tea_probs = F.softmax(tea_logits, dim=1)
                        max_probs, pseudo_labels = tea_probs.max(dim=1)
                        confident_mask = max_probs > self.confidence_threshold
                        confident_feat = vec_t[confident_mask]
                        confident_labels = pseudo_labels[confident_mask]

                    if confident_mask.sum() > 0:
                        loss_tgt = self.cross_entropy(self.C2(confident_feat), confident_labels)
                    else:
                        loss_tgt = torch.tensor(0.0, device=self.device)

                    loss_enc = loss_cls + loss_adv + self.teacher_weight * loss_tgt
                    loss_enc.backward()
                    self.optimizer_enc.step()

                    with torch.no_grad():
                        for t_p, s_p in zip(self.teacher_F.parameters(), self.F.parameters()):
                            t_p.data = self.ema_alpha * t_p.data + (1 - self.ema_alpha) * s_p.data
                        for t_p, s_p in zip(self.teacher_C.parameters(), self.C2.parameters()):
                            t_p.data = self.ema_alpha * t_p.data + (1 - self.ema_alpha) * s_p.data

                    epoch_loss['Source Classifier'] += loss_cls.item()
                    epoch_loss['Discriminator'] += loss_disc.item()
                    epoch_loss['Adversarial'] += loss_adv.item()
                    if confident_mask.sum() > 0:
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

                # ================================================
                # Step 1: Train Discriminator
                # ================================================
                self.optimizer_disc.zero_grad()
                self.optimizer_enc.zero_grad()

                vec_s, seq_s = self.F(source_data)
                vec_t, seq_t = self.F(target_data)

                # Detach features for discriminator training
                seq_concat = torch.cat((seq_s.detach(), seq_t.detach()), dim=0)
                disc_pred = self.discriminator(seq_concat).squeeze()

                # Domain labels: source = 1, target = 0
                label_src = torch.ones(seq_s.size(0), device=self.device)
                label_tgt = torch.zeros(seq_t.size(0), device=self.device)
                label_concat = torch.cat((label_src, label_tgt), dim=0)

                loss_disc = self.bce_loss(disc_pred, label_concat)
                loss_disc.backward()
                self.optimizer_disc.step()

                # Discriminator accuracy
                disc_acc = (
                    (torch.sigmoid(disc_pred) > 0.5).float() == label_concat
                ).float().mean()

                # ================================================
                # Step 2: Train Target Encoder (adversarial + self-training)
                # ================================================
                self.optimizer_disc.zero_grad()
                self.optimizer_enc.zero_grad()

                # Re-forward (features were detached above)
                vec_s, seq_s = self.F(source_data)
                vec_t, seq_t = self.F(target_data)

                # --- Source classification loss ---
                logits_s = self.C1(vec_s)
                loss_cls = self.cross_entropy(logits_s, source_labels)
                epoch_acc['Source C1'] += utils.get_accuracy(logits_s, source_labels)

                # --- Adversarial loss: fool the discriminator ---
                disc_pred_t = self.discriminator(seq_t).squeeze()
                label_adv = torch.ones(seq_t.size(0), device=self.device)  # target wants to be classified as source
                loss_adv = self.bce_loss(disc_pred_t, label_adv)

                # --- Mean Teacher: generate pseudo-labels ---
                with torch.no_grad():
                    tea_vec_t, _ = self.teacher_F(target_data)
                    tea_logits = self.teacher_C(tea_vec_t)
                    tea_probs = F.softmax(tea_logits, dim=1)
                    max_probs, pseudo_labels = tea_probs.max(dim=1)

                    # Only keep confident predictions
                    confident_mask = max_probs > self.confidence_threshold
                    confident_feat = vec_t[confident_mask]
                    confident_labels = pseudo_labels[confident_mask]

                # --- Target self-training loss ---
                if confident_mask.sum() > 0:
                    logits_t = self.C2(confident_feat)
                    loss_tgt = self.cross_entropy(logits_t, confident_labels)
                else:
                    loss_tgt = torch.tensor(0.0, device=self.device)

                # --- Total encoder loss ---
                loss_enc = loss_cls + loss_adv + self.teacher_weight * loss_tgt

                loss_enc.backward()
                self.optimizer_enc.step()

                # --- EMA update Mean Teacher ---
                with torch.no_grad():
                    for t_p, s_p in zip(self.teacher_F.parameters(), self.F.parameters()):
                        t_p.data = self.ema_alpha * t_p.data + (1 - self.ema_alpha) * s_p.data
                    for t_p, s_p in zip(self.teacher_C.parameters(), self.C2.parameters()):
                        t_p.data = self.ema_alpha * t_p.data + (1 - self.ema_alpha) * s_p.data

                # Record losses
                epoch_loss['Source Classifier'] += loss_cls.item()
                epoch_loss['Discriminator'] += loss_disc.item()
                epoch_loss['Adversarial'] += loss_adv.item()
                if confident_mask.sum() > 0:
                    epoch_loss['Target Self-Training'] += loss_tgt.item()
                epoch_acc['Discriminator'] += disc_acc.item()

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

            # ---- Log results ----
            logging.info(f"Epoch {epoch}: Val-Acc={new_acc:.4f}, Val-F1={new_f1:.4f}")
            logging.info("Best so far -> "
                         f"Acc: Epoch {best_acc_epoch} (Acc={best_acc:.4f}, F1={best_acc_f1:.4f}) | "
                         f"F1: Epoch {best_f1_epoch} (Acc={best_f1_acc:.4f}, F1={best_f1:.4f})")

            if self.lr_scheduler_enc is not None:
                self.lr_scheduler_enc.step()
            if self.lr_scheduler_disc is not None:
                self.lr_scheduler_disc.step()

        # ---- Training Summary ----
        logging.info('=' * 70)
        logging.info('Training finished! Best results summary:')
        logging.info(f'Best Accuracy: {best_acc:.4f} (Epoch {best_acc_epoch}, F1={best_acc_f1:.4f})')
        logging.info(f'Best F1-score: {best_f1:.4f} (Epoch {best_f1_epoch}, Acc={best_f1_acc:.4f})')
        logging.info('=' * 70)

    # ================== Validation ==================
    @torch.no_grad()
    def test(self):
        self.F.eval()
        self.C2.eval()
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

            vec_t, _ = self.F(target_data)
            logits = self.C2(vec_t)
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
