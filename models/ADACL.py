'''
Paper: Zhang, Y., Ren, Z., Zhou, S. and Yu, T., 2020. Adversarial domain adaptation with
    classifier alignment for cross-domain intelligent fault diagnosis of multiple source domains.
    Measurement Science and Technology, 32(3), p.035102.
Note: The code is reproduced according to the paper. If there is a performance loss compared to the paper,
    please point out the mistakes or provide the source code.
Author: Feaxure
'''
import torch
import logging
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

import utils
import model_base
from train_utils import InitTrain

##original
# class Trainset(InitTrain):
#
#     def __init__(self, args):
#         super(Trainset, self).__init__(args)
#         output_size = 2560
#         self.discriminator = model_base.ClassifierMLP(input_size=output_size, output_size=(self.num_source+1),
#                         dropout=args.dropout, last=None).to(self.device)
#         self.grl = utils.GradientReverseLayer()
#         self.G = model_base.FeatureExtractor(in_channel=1).to(self.device)
#         self.Cs = nn.ModuleList([model_base.ClassifierMLP(input_size=output_size, output_size=args.num_classes,
#                                                           dropout=args.dropout, last=None) \
#                                                           for _ in range(self.num_source)]).to(self.device)
#         self._init_data()
#
#     def save_model(self):
#         torch.save({
#             'G': self.G.state_dict(),
#             'Cs': self.Cs.state_dict()
#             }, self.args.save_path + '.pth')
#         logging.info('Model saved to {}'.format(self.args.save_path + '.pth'))
#
#     def load_model(self):
#         logging.info('Loading model from {}'.format(self.args.load_path))
#         ckpt = torch.load(self.args.load_path)
#         self.G.load_state_dict(ckpt['G'])
#         self.Cs.load_state_dict(ckpt['Cs'])
#
#     def train(self):
#         args = self.args
#         src = args.source_name
#
#         self.optimizer = self._get_optimizer([self.G, self.Cs, self.discriminator])
#         self.lr_scheduler = self._get_lr_scheduler(self.optimizer)
#
#         best_acc = 0.0
#         best_epoch = 0
#
#         for epoch in range(1, args.max_epoch+1):
#             logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-'*5)
#
#             # Update the learning rate
#             if self.lr_scheduler is not None:
#                 logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))
#
#             # Each epoch has a training and val phase
#             epoch_acc = defaultdict(float)
#
#             # Set model to train mode or evaluate mode
#             self.G.train()
#             self.Cs.train()
#             self.discriminator.train()
#             epoch_loss = defaultdict(float)
#             tradeoff = self._get_tradeoff(args.tradeoff, epoch)
#
#             num_iter = len(self.dataloaders['train'])
#             for i in tqdm(range(num_iter), ascii=True):
#                 target_data, target_labels = utils.get_next_batch(self.dataloaders,
#                 						 self.iters, 'train', self.device)
#                 source_data, source_labels, src_idx = utils.get_next_batch(self.dataloaders,
#             						     self.iters, src[int(i%len(args.source_name))], self.device, return_idx=True)
#                 if args.train_mode == 'multi_source':
#                     src_idx = src_idx[0]
#                 else:
#                     src_idx = 0
#                 # forward
#                 self.optimizer.zero_grad()
#
#                 batch_size = source_data.shape[0]
#                 data = torch.cat((source_data, target_data), dim=0)
#
#                 f = self.G(data)
#                 f_s, f_t = f.chunk(2, dim=0)
#                 y_s = self.Cs[src_idx](f_s)
#                 y_t = [cl(f_t) for cl in self.Cs]
#
#                 loss_c = F.cross_entropy(y_s, source_labels)
#
#                 labels_dm = torch.concat((torch.full((batch_size,), src_idx+1, dtype=torch.long),
#                                           torch.zeros(batch_size, dtype=torch.long)), dim=0).to(self.device)
#                 feat = self.grl(torch.concat((f_s, f_t), dim=0))
#                 logits_dm = self.discriminator(feat)
#                 loss_d = F.cross_entropy(logits_dm, labels_dm)
#
#                 logits_tgt = [F.softmax(t, dim=1) for t in y_t]
#                 loss_l1 = 0.0
#                 for k in range(self.num_source - 1):
#                     for j in range(k+1, self.num_source):
#                         # We use mean value of this result, even though the sum value is used in the paper.
#                         loss_l1 += torch.abs(logits_tgt[k] - logits_tgt[j]).mean()
#                 loss_l1 /= self.num_source
#
#                 loss = loss_c + tradeoff[0] * loss_d + tradeoff[1] * loss_l1
#
#                 epoch_acc['Source Data']  += utils.get_accuracy(y_s, source_labels)
#                 epoch_acc['Discriminator']  += utils.get_accuracy(logits_dm, labels_dm)
#
#                 epoch_loss['Source Classifier'] += loss_c
#                 epoch_loss['Discriminator'] += loss_d
#                 epoch_loss['L1'] += loss_l1
#
#                 # backward
#                 loss.backward()
#                 self.optimizer.step()
#
#             # Print the train and val information via each epoch
#             for key in epoch_loss.keys():
#                 logging.info('Train-Loss {}: {:.4f}'.format(key, epoch_loss[key]/num_iter))
#             for key in epoch_acc.keys():
#                 logging.info('Train-Acc {}: {:.4f}'.format(key, epoch_acc[key]/num_iter))
#
#             # log the best model according to the val accuracy
#             new_acc = self.test()
#             if new_acc >= best_acc:
#                 best_acc = new_acc
#                 best_epoch = epoch
#             logging.info("The best model epoch {}, val-acc {:.4f}".format(best_epoch, best_acc))
#
#             if self.lr_scheduler is not None:
#                 self.lr_scheduler.step()
#
#     def test(self):
#         self.G.eval()
#         self.Cs.eval()
#         acc = 0.0
#         iters = iter(self.dataloaders['val'])
#         num_iter = len(iters)
#         with torch.no_grad():
#             for i in tqdm(range(num_iter), ascii=True):
#                 target_data, target_labels, _ = next(iters)
#                 target_data, target_labels = target_data.to(self.device), target_labels.to(self.device)
#                 feat_tgt = self.G(target_data)
#                 logits_tgt = [cl(feat_tgt) for cl in self.Cs]
#                 logits_tgt = [F.softmax(data, dim=1) for data in logits_tgt]
#
#                 pred = torch.zeros((logits_tgt[0].shape)).to(self.device)
#                 for j in range(self.num_source):
#                     pred += logits_tgt[j]
#                 acc += utils.get_accuracy(pred, target_labels)
#         acc /= num_iter
#         logging.info('Val-Acc Target Data: {:.4f}'.format(acc))
#         return acc

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from tqdm import tqdm


class Trainset(InitTrain):

    def __init__(self, args):
        super(Trainset, self).__init__(args)
        output_size = 2560
        self.discriminator = model_base.ClassifierMLP(
            input_size=output_size,
            output_size=(self.num_source + 1),
            dropout=args.dropout,
            last=None
        ).to(self.device)
        self.grl = utils.GradientReverseLayer()
        self.G = model_base.FeatureExtractor(in_channel=1).to(self.device)
        self.Cs = nn.ModuleList([
            model_base.ClassifierMLP(
                input_size=output_size,
                output_size=args.num_classes,
                dropout=args.dropout,
                last=None
            ) for _ in range(self.num_source)
        ]).to(self.device)
        self._init_data()

        # 自动检测任务类型并设置F1参数
        self.num_classes = args.num_classes
        self.f1_average = self._get_f1_average()
        logging.info(f"Task: {self.num_classes}-class classification | F1 average: '{self.f1_average}'")

    def _get_f1_average(self):
        """根据类别数自动选择F1计算方式"""
        if self.num_classes == 2:
            return 'binary'  # 二分类专用
        elif self.num_classes > 2:
            return 'weighted'  # 多分类推荐
        else:
            return 'micro'  # 单分类（理论上不会发生）

    def save_model_acc(self):
        # 保存最优Acc模型
        save_path = self.args.save_path + '_best_acc.pth'
        torch.save({
            'G': self.G.state_dict(),
            'Cs': self.Cs.state_dict(),
            'discriminator': self.discriminator.state_dict()
        }, save_path)
        logging.info('ACC Model saved to {}'.format(save_path))

    def save_model_f1(self):
        # 保存最优F1模型
        save_path = self.args.save_path + '_best_f1.pth'
        torch.save({
            'G': self.G.state_dict(),
            'Cs': self.Cs.state_dict(),
            'discriminator': self.discriminator.state_dict()
        }, save_path)
        logging.info('F1 Model saved to {}'.format(save_path))

    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path)
        self.G.load_state_dict(ckpt['G'])
        self.Cs.load_state_dict(ckpt['Cs'])
        if 'discriminator' in ckpt:
            self.discriminator.load_state_dict(ckpt['discriminator'])

    def train(self):
        args = self.args
        src = args.source_name

        self.optimizer = self._get_optimizer([self.G, self.Cs, self.discriminator])
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)

        # 初始化最优指标记录
        best_acc = 0.0
        best_acc_epoch = 0
        best_acc_f1_at_epoch = 0.0

        best_f1 = 0.0
        best_f1_epoch = 0
        best_f1_acc_at_epoch = 0.0

        for epoch in range(1, args.max_epoch + 1):
            self.current_epoch = epoch
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-' * 5)

            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))

            epoch_acc = defaultdict(float)

            self.G.train()
            self.Cs.train()
            self.discriminator.train()
            epoch_loss = defaultdict(float)
            tradeoff = self._get_tradeoff(args.tradeoff, epoch)

            num_iter = len(self.dataloaders['train'])
            if num_iter == 0:
                logging.error("Training dataloader is empty!")
                break

            for i in tqdm(range(num_iter), ascii=True):
                try:
                    target_data, target_labels = utils.get_next_batch(
                        self.dataloaders, self.iters, 'train', self.device
                    )
                    source_data, source_labels, src_idx = utils.get_next_batch(
                        self.dataloaders, self.iters,
                        src[int(i % len(args.source_name))], self.device, return_idx=True
                    )
                except StopIteration:
                    break

                if args.train_mode == 'multi_source':
                    src_idx = src_idx[0]
                else:
                    src_idx = 0

                self.optimizer.zero_grad()

                batch_size = source_data.shape[0]
                data = torch.cat((source_data, target_data), dim=0)

                f = self.G(data)
                f_s, f_t = f.chunk(2, dim=0)
                y_s = self.Cs[src_idx](f_s)
                y_t = [cl(f_t) for cl in self.Cs]

                loss_c = F.cross_entropy(y_s, source_labels)

                labels_dm = torch.concat((
                    torch.full((batch_size,), src_idx + 1, dtype=torch.long),
                    torch.zeros(batch_size, dtype=torch.long)
                ), dim=0).to(self.device)

                feat = self.grl(torch.concat((f_s, f_t), dim=0))
                logits_dm = self.discriminator(feat)
                loss_d = F.cross_entropy(logits_dm, labels_dm)

                # ================== 修复点 ==================
                loss_l1 = torch.tensor(0.0, device=self.device)

                if self.num_source > 1:
                    logits_tgt = [F.softmax(t, dim=1) for t in y_t]
                    for k in range(self.num_source - 1):
                        for j in range(k + 1, self.num_source):
                            loss_l1 += torch.abs(logits_tgt[k] - logits_tgt[j]).mean()
                    loss_l1 = loss_l1 / self.num_source
                # ============================================

                loss = loss_c + tradeoff[0] * loss_d + tradeoff[1] * loss_l1

                epoch_acc['Source Data'] += utils.get_accuracy(y_s, source_labels)
                epoch_acc['Discriminator'] += utils.get_accuracy(logits_dm, labels_dm)

                epoch_loss['Source Classifier'] += loss_c.item()
                epoch_loss['Discriminator'] += loss_d.item()
                epoch_loss['L1'] += loss_l1.item()  # ✅ 现在安全

                loss.backward()
                self.optimizer.step()

            for key in epoch_loss.keys():
                logging.info('Train-Loss {}: {:.4f}'.format(key, epoch_loss[key] / max(num_iter, 1)))
            for key in epoch_acc.keys():
                logging.info('Train-Acc {}: {:.4f}'.format(key, epoch_acc[key] / max(num_iter, 1)))

            new_acc, new_f1 = self.test()

            if new_acc >= best_acc:
                best_acc = new_acc
                best_acc_epoch = epoch
                best_acc_f1_at_epoch = new_f1
                self.save_model_acc()

            if new_f1 >= best_f1:
                best_f1 = new_f1
                best_f1_epoch = epoch
                best_f1_acc_at_epoch = new_acc
                self.save_model_f1()

            logging.info(f"Epoch {epoch}: Val-Acc={new_acc:.4f}, Val-F1={new_f1:.4f}")
            logging.info("Best so far → "
                         f"Acc: Epoch {best_acc_epoch} (Acc={best_acc:.4f}, F1={best_acc_f1_at_epoch:.4f}) | "
                         f"F1: Epoch {best_f1_epoch} (Acc={best_f1_acc_at_epoch:.4f}, F1={best_f1:.4f})")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        logging.info('=' * 70)
        logging.info('Training finished! Best results summary:')
        logging.info(f'Best Accuracy: {best_acc:.4f} (Epoch {best_acc_epoch}, F1={best_acc_f1_at_epoch:.4f})')
        logging.info(f'Best F1-score: {best_f1:.4f} (Epoch {best_f1_epoch}, Acc={best_f1_acc_at_epoch:.4f})')
        logging.info('=' * 70)

    def test(self):
        self.G.eval()
        self.Cs.eval()
        self.discriminator.eval()
        all_preds = []
        all_labels = []

        # 检查验证集是否存在
        if 'val' not in self.dataloaders or self.dataloaders['val'] is None:
            logging.warning("No validation dataloader found!")
            return 0.0, 0.0

        try:
            iters = iter(self.dataloaders['val'])
            num_iter = len(self.dataloaders['val'])
        except:
            logging.warning("Failed to create validation iterator!")
            return 0.0, 0.0

        # 检查验证集是否为空
        if num_iter == 0:
            logging.warning("Validation dataloader is empty!")
            return 0.0, 0.0

        with torch.no_grad():
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

                feat_tgt = self.G(target_data)
                logits_tgt = [cl(feat_tgt) for cl in self.Cs]
                logits_tgt = [F.softmax(data, dim=1) for data in logits_tgt]

                # 平均多个分类器的预测
                pred_probs = torch.zeros_like(logits_tgt[0])
                for j in range(self.num_source):
                    pred_probs += logits_tgt[j]
                pred_probs /= self.num_source

                pred_labels = torch.argmax(pred_probs, dim=1)

                all_preds.extend(pred_labels.cpu().numpy())
                all_labels.extend(target_labels.cpu().numpy())

        # 检查是否有有效预测
        if len(all_preds) == 0:
            logging.warning("No predictions made during validation!")
            return 0.0, 0.0

        # 转换为numpy数组
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # 计算准确率
        acc = accuracy_score(all_labels, all_preds)

        # 处理单类别特殊情况
        unique_labels = np.unique(all_labels)
        if len(unique_labels) < 2:
            f1 = 0.0
            logging.warning(f"Only {len(unique_labels)} class(es) present in validation set. F1 score is not defined.")
        else:
            try:
                # 自动适配二分类/多分类
                if self.num_classes == 2:
                    # 二分类：自动检测正类标签
                    pos_label = 1 if 1 in unique_labels else unique_labels[-1]
                    f1 = f1_score(all_labels, all_preds, average='binary', pos_label=pos_label)
                else:
                    # 多分类：使用预定义的平均方式
                    f1 = f1_score(all_labels, all_preds, average=self.f1_average)
            except Exception as e:
                logging.warning(f"Error calculating F1: {e}")
                # 回退到micro平均
                f1 = f1_score(all_labels, all_preds, average='micro')

        logging.info('Val-Acc : {:.4f}'.format(acc))
        logging.info('Val-F1  : {:.4f}'.format(f1))

        return acc, f1