'''
Paper: Yu, Y., He, Y., Karimi, H.R., Gelman, L. and Cetin, A.E., 2024.
    A two-stage importance-aware subgraph convolutional network based on
    multi-source sensors for cross-domain fault diagnosis. Neural Networks.
Reference code: https://github.com/Polimi-YuYue/MSI2DA-Net
Note: Adapted for 1D vibration signal bearing fault diagnosis.

    Core innovations:
      1. Multi-Source Sensor Fusion — fuses features from multiple signal
         representations (time-domain, frequency-domain, time-frequency)
         to build a richer graph structure.
      2. Two-Stage Fault Diagnosis:
         - Stage 1 (MIL): Multi-Instance Learning for sensor importance weighting
         - Stage 2 (I²SGCNs): Importance-aware Subgraph Convolutional Networks
           for fine-grained cross-domain fault diagnosis.
      3. Feature-based + Importance-based joint graph mining.

    Compatible with: python train.py --model_name MSI2DA_Net --source CWRU_0 --target CWRU_1
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
# 1. Multi-Instance Learning for Sensor Importance
# ======================================================================
class MultiInstanceAttention(nn.Module):
    """
    Multi-Instance Learning (MIL) attention: given multiple sensor views
    (time-domain raw, frequency-domain, envelope spectrum), learn per-sensor
    importance weights via attention pooling.
    """
    def __init__(self, input_dim, hidden_dim=128):
        super(MultiInstanceAttention, self).__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, features_list):
        """
        Args:
            features_list: list of (B, D) tensors, one per sensor view
        Returns:
            fused: (B, D) weighted sum of features
            weights: (B, K) sensor importance weights
        """
        # Stack: (B, K, D) where K = number of views
        stacked = torch.stack(features_list, dim=1)  # (B, K, D)

        B, K, D = stacked.shape

        # Flatten for attention
        flat = stacked.reshape(B * K, D)

        A_V = self.attention_V(flat)  # (B*K, H)
        A_U = self.attention_U(flat)  # (B*K, H)
        A = self.attention_weights(A_V * A_U)  # (B*K, 1)
        A = A.reshape(B, K)  # (B, K)

        # Softmax across views
        weights = F.softmax(A, dim=1)  # (B, K)

        # Weighted sum
        fused = (stacked * weights.unsqueeze(2)).sum(dim=1)  # (B, D)

        return fused, weights


# ======================================================================
# 2. Importance-Aware Subgraph Convolutional Network (I²SGCN)
# ======================================================================
class ImportanceAwareSubgraphGCN(nn.Module):
    """
    Builds two parallel subgraphs:
      - Feature-based graph: k-NN on features
      - Importance-based graph: weighted by sensor importance scores

    Then jointly mines effective node relationships.
    """
    def __init__(self, input_dim, hidden_dim=256, output_dim=128, dropout=0.):
        super(ImportanceAwareSubgraphGCN, self).__init__()

        # Feature-based graph convolution
        self.gcn_feat = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # Importance-based graph convolution
        self.gcn_imp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )

    def _build_feature_graph(self, x):
        """Build k-NN graph based on feature similarity."""
        B = x.size(0)
        device = x.device
        dist = torch.cdist(x, x, p=2)
        _, indices = dist.topk(min(8, B), dim=1, largest=False)
        adj = torch.zeros(B, B, device=device)
        for i in range(B):
            adj[i, indices[i]] = 1.0
        adj = (adj + adj.T).clamp(0, 1)
        D = adj.sum(dim=1) + 1e-6
        adj_norm = adj / D.unsqueeze(1)
        return adj_norm

    def _build_importance_graph(self, x, importance_weights):
        """
        Build importance-weighted graph:
        edge weight = similarity * (importance_i + importance_j) / 2
        """
        B = x.size(0)
        device = x.device

        # Cosine similarity matrix
        x_norm = F.normalize(x, dim=1)
        sim = x_norm @ x_norm.T  # (B, B)

        # Importance weighting
        imp = importance_weights.mean(dim=1)  # (B,)
        imp_matrix = (imp.unsqueeze(0) + imp.unsqueeze(1)) / 2  # (B, B)

        adj = sim * imp_matrix
        adj = adj.clamp(min=0)

        # Normalize
        D = adj.sum(dim=1) + 1e-6
        adj_norm = adj / D.unsqueeze(1)

        return adj_norm

    def forward(self, x, importance_weights):
        """
        Args:
            x: (B, D) features
            importance_weights: (B, K) sensor importance
        Returns:
            fused: (B, output_dim)
        """
        # Feature-based graph
        adj_feat = self._build_feature_graph(x)
        g_feat = self.gcn_feat(adj_feat @ x)

        # Importance-based graph
        adj_imp = self._build_importance_graph(x, importance_weights)
        g_imp = self.gcn_imp(adj_imp @ x)

        # Fuse
        fused = self.fusion(torch.cat([g_feat, g_imp], dim=1))
        return fused


# ======================================================================
# 3. Multi-Source Feature Extraction
# ======================================================================
class MultiSourceEncoder(nn.Module):
    """
    Extract features from multiple signal representations:
      - Raw time-domain (via FeatureExtractor)
      - Frequency-domain (FFT spectrum)
      - Time-frequency (via small CNN on Mel-like representation)
    """
    def __init__(self, signal_length=1024, dropout=0.):
        super(MultiSourceEncoder, self).__init__()

        # Frequency-domain encoder
        self.freq_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(32),
            nn.Flatten(),
            nn.Linear(16 * 32, 128),
        )

        # Envelope encoder (Hilbert-like via abs of signal)
        self.env_encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=16, stride=4, padding=7),
            nn.BatchNorm1d(8), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(8 * 16, 128),
        )

    def forward(self, x_time, x):
        """
        Args:
            x_time: (B, 2560) time-domain features (from FeatureExtractor)
            x: (B, 1, L) raw signal
        Returns:
            views: list of (B, 128) features from different sources
        """
        # Frequency domain
        x_fft = torch.abs(torch.fft.rfft(x.squeeze(1), dim=1))
        # Pad/truncate to consistent length
        target_len = 512
        if x_fft.size(1) < target_len:
            x_fft = F.pad(x_fft, (0, target_len - x_fft.size(1)))
        else:
            x_fft = x_fft[:, :target_len]
        x_fft = x_fft.unsqueeze(1)  # (B, 1, 512)
        f_freq = self.freq_encoder(x_fft)  # (B, 128)

        # Envelope (simulated)
        x_env = torch.abs(x)  # (B, 1, L)
        f_env = self.env_encoder(x_env)  # (B, 128)

        return [f_freq, f_env]  # Two additional views


# ======================================================================
# 4. Trainset — Standard Interface
# ======================================================================
class Trainset(InitTrain):
    """
    MSI2DA-Net: Multi-Source Importance-aware Subgraph Convolutional Network.

    Two-stage architecture:
      Stage 1: MIL attention → per-source importance weights
      Stage 2: I²SGCNs → importance-weighted graph convolution + classification

    Architecture:
      1. Feature Extractor F (time-domain) + MultiSourceEncoder (freq+env)
      2. MIL Attention → fuse sources with importance weights
      3. I²SGCN → feature-based + importance-based graph convolution
      4. Classifier C
      5. Domain adversarial alignment

    Loss = L_cls + λ_adv*L_adv + λ_balance*L_balance
    """

    def __init__(self, args):
        super(Trainset, self).__init__(args)

        base_dim = 2560
        view_dim = 128

        # ---------- 1. Time-domain Feature Extractor ----------
        self.F = model_base.FeatureExtractor(
            in_channel=1, dropout=args.dropout
        ).to(self.device)

        # ---------- 2. Multi-Source Encoder ----------
        self.ms_encoder = MultiSourceEncoder(
            signal_length=args.signal_size, dropout=args.dropout
        ).to(self.device)

        # ---------- 3. MIL Attention ----------
        self.mil_attn = MultiInstanceAttention(
            input_dim=view_dim, hidden_dim=64
        ).to(self.device)

        # ---------- 4. I²SGCN ----------
        self.isgcn = ImportanceAwareSubgraphGCN(
            input_dim=view_dim, hidden_dim=128, output_dim=128, dropout=args.dropout
        ).to(self.device)

        # ---------- 5. Classifier ----------
        self.C = model_base.ClassifierMLP(
            input_size=128, output_size=args.num_classes,
            dropout=args.dropout, last=None
        ).to(self.device)

        # ---------- 6. Domain Discriminator ----------
        self.train_mode = args.train_mode
        grl = utils.GradientReverseLayer()
        if self.train_mode == 'multi_source':
            self.domain_discri = nn.ModuleList([
                model_base.ClassifierMLP(
                    input_size=128, output_size=1,
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
                input_size=128, output_size=1,
                dropout=args.dropout, last='sigmoid'
            ).to(self.device)
            self.domain_adv = utils.DomainAdversarialLoss(self.domain_discri, grl=grl)
            self.domain_weight_module = None

        # ---------- 7. Hyperparameters ----------
        self.balance_weight = 0.1  # Importance balance regularization

        # ---------- 8. F1 Settings ----------
        self.num_classes = args.num_classes
        self.f1_average = self._get_f1_average()
        logging.info(f"MSI2DA-Net | {self.num_classes}-class | F1: '{self.f1_average}'")

        self._init_data()

    def _get_f1_average(self):
        if self.num_classes == 2: return 'binary'
        elif self.num_classes > 2: return 'weighted'
        else: return 'micro'

    def _extract_features(self, x_raw, f_time):
        """Full feature pipeline: multi-source → MIL → I²SGCN."""
        # Stage 1: Multi-source views
        views = self.ms_encoder(f_time, x_raw)
        views.append(f_time[:, :128])  # Add time-domain view (truncated)

        # MIL attention fusion
        fused, imp_weights = self.mil_attn(views)  # (B, 128), (B, K)

        # Stage 2: I²SGCN
        f_graph = self.isgcn(fused, imp_weights)  # (B, 128)

        return f_graph, imp_weights

    def _discriminator_state_dict(self):
        if self.train_mode == 'multi_source':
            return {f'disc_{i}': d.state_dict() for i, d in enumerate(self.domain_discri)}
        return self.domain_discri.state_dict()

    def save_model_acc(self):
        save_path = self.args.save_path + '_best_acc.pth'
        torch.save({
            'F': self.F.state_dict(), 'ms_encoder': self.ms_encoder.state_dict(),
            'mil_attn': self.mil_attn.state_dict(), 'isgcn': self.isgcn.state_dict(),
            'C': self.C.state_dict(), 'domain_discri': self._discriminator_state_dict(),
        }, save_path)
        logging.info('ACC Model saved to {}'.format(save_path))

    def save_model_f1(self):
        save_path = self.args.save_path + '_best_f1.pth'
        torch.save({
            'F': self.F.state_dict(), 'ms_encoder': self.ms_encoder.state_dict(),
            'mil_attn': self.mil_attn.state_dict(), 'isgcn': self.isgcn.state_dict(),
            'C': self.C.state_dict(), 'domain_discri': self._discriminator_state_dict(),
        }, save_path)
        logging.info('F1 Model saved to {}'.format(save_path))

    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path, map_location=self.device)
        for k in ['F', 'ms_encoder', 'mil_attn', 'isgcn', 'C']:
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
            self.F, self.ms_encoder, self.mil_attn, self.isgcn, self.C, self.domain_discri
        ])
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)

        best_acc = 0.0; best_acc_epoch = 0; best_acc_f1 = 0.0
        best_f1 = 0.0; best_f1_epoch = 0; best_f1_acc = 0.0

        for epoch in range(1, args.max_epoch + 1):
            self.current_epoch = epoch
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-' * 5)
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))

            for m in [self.F, self.ms_encoder, self.mil_attn, self.isgcn, self.C, self.domain_discri]: m.train()
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

                    f_time_t = self.F(target_data)
                    f_t, imp_t = self._extract_features(target_data, f_time_t)
                    loss_per_domain = torch.zeros(self.num_source).to(self.device)
                    loss_adapt_per_domain = torch.zeros(self.num_source).to(self.device)
                    for idx in range(self.num_source):
                        f_time_s = self.F(source_data[idx])
                        f_s, imp_s = self._extract_features(source_data[idx], f_time_s)
                        logits_s = self.C(f_s)
                        loss_cls = F.cross_entropy(logits_s, source_labels[idx])
                        epoch_acc['Source'] += utils.get_accuracy(logits_s, source_labels[idx]) / self.num_source

                        loss_adv, acc_d = self.domain_adv[idx](f_s, f_t)
                        epoch_acc['Discriminator'] += acc_d / self.num_source

                        num_views = imp_s.size(1)
                        loss_balance = ((imp_s.mean(dim=0) - 1.0 / num_views) ** 2).mean()

                        loss_per_domain[idx] = loss_cls
                        loss_adapt_per_domain[idx] = loss_adv + self.balance_weight * loss_balance

                    self.domain_weight_module.update(loss_per_domain, domain_idxes)
                    domain_weight = self.domain_weight_module.get_domain_weight(domain_idxes)
                    loss = (domain_weight * (loss_per_domain + tradeoff[0] * loss_adapt_per_domain)).sum()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.F.parameters()) + list(self.ms_encoder.parameters()) +
                        list(self.mil_attn.parameters()) + list(self.isgcn.parameters()) +
                        list(self.C.parameters()) +
                        [p for d in self.domain_discri for p in d.parameters()], max_norm=10.0)
                    self.optimizer.step()

                    epoch_loss['Classifier'] += (domain_weight * loss_per_domain).sum().item()
                    epoch_loss['Adversarial'] += (domain_weight * loss_adapt_per_domain).sum().item()
                    continue

                try:
                    target_data, _ = utils.get_next_batch(self.dataloaders, self.iters, 'train', self.device)
                    source_data, source_labels = utils.get_next_batch(self.dataloaders, self.iters, src, self.device)
                except StopIteration: break

                f_time_s = self.F(source_data); f_time_t = self.F(target_data)

                f_s, imp_s = self._extract_features(source_data, f_time_s)
                f_t, imp_t = self._extract_features(target_data, f_time_t)

                # === Classification ===
                logits_s = self.C(f_s)
                loss_cls = F.cross_entropy(logits_s, source_labels)
                epoch_acc['Source'] += utils.get_accuracy(logits_s, source_labels)

                # === Domain adversarial ===
                loss_adv, acc_d = self.domain_adv(f_s, f_t)
                epoch_acc['Discriminator'] += acc_d

                # === Importance balance (prevent collapse to single view) ===
                # Encourage uniform importance across views
                num_views = imp_s.size(1)
                loss_balance = ((imp_s.mean(dim=0) - 1.0 / num_views) ** 2).mean()

                # === Total Loss ===
                loss = loss_cls \
                    + tradeoff[0] * loss_adv \
                    + self.balance_weight * loss_balance

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.F.parameters()) + list(self.ms_encoder.parameters()) +
                    list(self.mil_attn.parameters()) + list(self.isgcn.parameters()) +
                    list(self.C.parameters()) + list(self.domain_discri.parameters()),
                    max_norm=10.0)
                self.optimizer.step()

                epoch_loss['Classifier'] += loss_cls.item()
                epoch_loss['Adversarial'] += loss_adv.item()
                epoch_loss['Importance Bal'] += loss_balance.item()

            for key in sorted(epoch_loss): logging.info('Train-Loss {}: {:.4f}'.format(key, epoch_loss[key] / max(num_iter, 1)))
            for key in sorted(epoch_acc): logging.info('Train-Acc {}: {:.4f}'.format(key, epoch_acc[key] / max(num_iter, 1)))

            new_acc, new_f1 = self.test()
            if new_acc >= best_acc: best_acc, best_acc_epoch, best_acc_f1 = new_acc, epoch, new_f1; self.save_model_acc()
            if new_f1 >= best_f1: best_f1, best_f1_epoch, best_f1_acc = new_f1, epoch, new_acc; self.save_model_f1()

            logging.info(f"Epoch {epoch}: Val-Acc={new_acc:.4f}, Val-F1={new_f1:.4f}")
            logging.info(f"Best → Acc: Ep{best_acc_epoch}({best_acc:.4f}) | F1: Ep{best_f1_epoch}({best_f1:.4f})")
            if self.lr_scheduler is not None: self.lr_scheduler.step()

        logging.info('=' * 70)
        logging.info(f'MSI2DA-Net Best: Acc={best_acc:.4f} (Ep{best_acc_epoch}) | F1={best_f1:.4f} (Ep{best_f1_epoch})')
        logging.info('=' * 70)

    @torch.no_grad()
    def test(self):
        for m in [self.F, self.ms_encoder, self.mil_attn, self.isgcn, self.C, self.domain_discri]: m.eval()
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
            f_time = self.F(td)
            f, _ = self._extract_features(td, f_time)
            logits = self.C(f)
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
