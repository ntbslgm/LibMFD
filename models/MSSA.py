'''
Paper: Tian, J., Han, D., Li, M. and Shi, P., 2022. A multi-source information transfer 
    learning method with subdomain adaptation for cross-domain fault diagnosis.
    Knowledge-Based Systems, 243, p.108466.
Note: The code has been developed in accordance with the methodologies described in the paper.
    Should there be any discrepancies in performance relative to the published results, we welcome
    feedback on potential errors or provision of source code. During our testing, we observed that
    certain feature extractors and the LMMD technique outlined in the paper may lead to a decrease
    in performance. Those elements can refer to the sections of the code that are commented out.
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

#original
# def one_hot(x, class_count):
#     return torch.eye(class_count)[x,:]
#
#
# class Trainset(InitTrain):
#
#     def __init__(self, args):
#         super(Trainset, self).__init__(args)
#         output_size = 2560
#         self.G_shared = model_base.FeatureExtractor(in_channel=1).to(self.device)
#         '''
#         # Specific feature extractors defined in the paper will not be used.
#         self.Gs_specific = nn.ModuleList([nn.Sequential(
#                                                     nn.Dropout(args.dropout),
#                                                     nn.Linear(output_size, output_size),
#                                                     nn.ReLU()) \
#                                                     for _ in range(self.num_source)]).to(self.device)
#         '''
#         self.Cs = nn.ModuleList([model_base.ClassifierMLP(input_size=output_size, output_size=args.num_classes,
#                                                           dropout=args.dropout, last=None) \
#                                                           for _ in range(self.num_source)]).to(self.device)
#         self.mkmmd = utils.MultipleKernelMaximumMeanDiscrepancy(
#                     kernels=[utils.GaussianKernel(alpha=2 ** k) for k in range(-3, 2)])
#
#     def save_model(self):
#         torch.save({
#             'G_shared': self.G_shared.state_dict(),
#             # 'Gs_specific': self.Gs_specific.state_dict(),
#             'Cs': self.Cs.state_dict()
#             }, self.args.save_path + '.pth')
#         logging.info('Model saved to {}'.format(self.args.save_path + '.pth'))
#
#     def load_model(self):
#         logging.info('Loading model from {}'.format(self.args.load_path))
#         ckpt = torch.load(self.args.load_path)
#         self.G_shared.load_state_dict(ckpt['G_shared'])
#         # self.Gs_specific.load_state_dict(ckpt['Gs_specific'])
#         self.Cs.load_state_dict(ckpt['Cs'])
#
#     def train(self):
#         args = self.args
#         self._init_data()
#         src = args.source_name
#
#         '''
#         # Specific feature extractors defined in the paper will not be used.
#         self.optimizer = self._get_optimizer([self.G_shared, self.Gs_specific, self.Cs])
#         '''
#         self.optimizer = self._get_optimizer([self.G_shared, self.Cs])
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
#             self.G_shared.train()
#             '''
#             # Specific feature extractors defined in the paper will not be used.
#             self.Gs_specific.train()
#             '''
#             self.Cs.train()
#             epoch_loss = defaultdict(float)
#             tradeoff = self._get_tradeoff(args.tradeoff, epoch)
#
#             num_iter = len(self.dataloaders['train'])
#             for i in tqdm(range(num_iter), ascii=True):
#                 target_data, target_labels = utils.get_next_batch(self.dataloaders,
#                 						 self.iters, 'train', self.device)
#                 source_data, source_labels = [], []
#                 if args.train_mode == 'source_combine':
#                     source_data_item, source_labels_item = utils.get_next_batch(self.dataloaders,
#                 						     self.iters, src, self.device)
#                     source_data.append(source_data_item)
#                     source_labels.append(source_labels_item)
#                 else:
#                     for idx in range(self.num_source):
#                         source_data_item, source_labels_item = utils.get_next_batch(self.dataloaders,
#                     						     self.iters, src[idx], self.device)
#                         source_data.append(source_data_item)
#                         source_labels.append(source_labels_item)
#                 # forward
#                 self.optimizer.zero_grad()
#                 data = torch.cat(source_data+[target_data], dim=0)
#                 f_shared = self.G_shared(data)
#                 f = f_shared.chunk(self.num_source+1, dim=0)
#
#                 '''
#                 # Specific feature extractors defined in the paper will not be used.
#                 f = [self.Gs_specific[j](f[j]) for j in range(self.num_source)]
#                 '''
#
#                 '''
#                 # LMMD defined in the paper will not be used.
#                 f_specific_t = [fs(f_shared_t) for fs in self.Gs_specific]
#                 y_s = [self.Cs[j](f_specific[j]) for j in range(self.num_source)]
#                 '''
#
#                 y_s = [self.Cs[j](f[j]) for j in range(self.num_source)]
#                 loss_cls = 0.0
#                 for j in range(self.num_source):
#                     loss_cls += F.cross_entropy(y_s[j], source_labels[j])
#                 loss_sum_mmd = 0.0
#                 for k in range(self.num_source):
#                     loss_sum_mmd += self.mkmmd(f[k], f[-1])
#
#                 '''
#                 # LMMD defined in the paper will not be used.
#                 y_t = [self.Cs[j](f_specific_t[j]) for j in range(self.num_source)]
#                 y_t = [F.softmax(data, dim=1) for data in y_t]
#                 for k in range(self.num_source):
#                     oh_label = one_hot(source_labels[k], args.num_classes)
#                     for j in range(args.num_classes):
#                         w_src = oh_label[:, j].view(-1, 1).to(self.device)
#                         w_tgt = y_t[k][:, j].view(-1, 1).to(self.device)
#                         loss_sum_mmd += self.mkmmd(w_src*f_specific[k], w_tgt*f_specific_t[k])
#                 '''
#
#                 loss = loss_cls + tradeoff[0] * loss_sum_mmd
#
#                 for j in range(self.num_source):
#                     epoch_acc['Source Data %d'%j] += utils.get_accuracy(y_s[j], source_labels[j])
#
#                 epoch_loss['Source Classifier'] += loss_cls
#                 epoch_loss['MMD'] += loss_sum_mmd
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
#         self.G_shared.eval()
#         '''
#         # Specific feature extractors defined in the paper will not be used.
#         self.Gs_specific.eval()
#         '''
#         self.Cs.eval()
#         acc = 0.0
#         iters = iter(self.dataloaders['val'])
#         num_iter = len(iters)
#         with torch.no_grad():
#             for i in tqdm(range(num_iter), ascii=True):
#                 target_data, target_labels, _ = next(iters)
#                 target_data, target_labels = target_data.to(self.device), target_labels.to(self.device)
#                 feat_tgt = self.G_shared(target_data)
#                 '''
#                 # Specific feature extractors defined in the paper will not be used.
#                 feat_tgt = [self.Gs_specific[j](feat_tgt) for j in range(self.num_source)]
#                 logits_tgt = [self.Cs[j](feat_tgt[j]) for j in range(self.num_source)]
#                 '''
#                 logits_tgt = [self.Cs[j](feat_tgt) for j in range(self.num_source)]
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

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from tqdm import tqdm


def one_hot(x, class_count):
    return torch.eye(class_count)[x, :]


class Trainset(InitTrain):

    def __init__(self, args):
        super(Trainset, self).__init__(args)
        output_size = 2560
        self.G_shared = model_base.FeatureExtractor(in_channel=1).to(self.device)
        '''
        # Specific feature extractors defined in the paper will not be used.
        self.Gs_specific = nn.ModuleList([nn.Sequential(
                                                    nn.Dropout(args.dropout),
                                                    nn.Linear(output_size, output_size),
                                                    nn.ReLU()) \
                                                    for _ in range(self.num_source)]).to(self.device)
        '''
        self.Cs = nn.ModuleList([model_base.ClassifierMLP(
            input_size=output_size,
            output_size=args.num_classes,
            dropout=args.dropout,
            last=None
        ) for _ in range(self.num_source)]).to(self.device)
        self.mkmmd = utils.MultipleKernelMaximumMeanDiscrepancy(
            kernels=[utils.GaussianKernel(alpha=2 ** k) for k in range(-3, 2)])

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
            'G_shared': self.G_shared.state_dict(),
            'Cs': self.Cs.state_dict()
        }, save_path)
        logging.info('ACC Model saved to {}'.format(save_path))

    def save_model_f1(self):
        # 保存最优F1模型
        save_path = self.args.save_path + '_best_f1.pth'
        torch.save({
            'G_shared': self.G_shared.state_dict(),
            'Cs': self.Cs.state_dict()
        }, save_path)
        logging.info('F1 Model saved to {}'.format(save_path))

    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path)
        self.G_shared.load_state_dict(ckpt['G_shared'])
        # self.Gs_specific.load_state_dict(ckpt['Gs_specific'])
        self.Cs.load_state_dict(ckpt['Cs'])

    def train(self):
        args = self.args
        self._init_data()
        src = args.source_name

        '''
        # Specific feature extractors defined in the paper will not be used.
        self.optimizer = self._get_optimizer([self.G_shared, self.Gs_specific, self.Cs])
        '''
        self.optimizer = self._get_optimizer([self.G_shared, self.Cs])
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)

        # 初始化最优指标记录
        best_acc = 0.0
        best_acc_epoch = 0
        best_acc_f1_at_epoch = 0.0  # 最佳Acc对应的F1

        best_f1 = 0.0
        best_f1_epoch = 0
        best_f1_acc_at_epoch = 0.0  # 最佳F1对应的Acc

        for epoch in range(1, args.max_epoch + 1):
            self.current_epoch = epoch
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-' * 5)

            # Update the learning rate
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))

            # Each epoch has a training and val phase
            epoch_acc = defaultdict(float)

            # Set model to train mode or evaluate mode
            self.G_shared.train()
            '''
            # Specific feature extractors defined in the paper will not be used.
            self.Gs_specific.train()
            '''
            self.Cs.train()
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
                except StopIteration:
                    break

                source_data, source_labels = [], []
                if args.train_mode == 'source_combine':
                    try:
                        source_data_item, source_labels_item = utils.get_next_batch(
                            self.dataloaders, self.iters, src, self.device
                        )
                        source_data.append(source_data_item)
                        source_labels.append(source_labels_item)
                    except StopIteration:
                        break
                else:
                    for idx in range(self.num_source):
                        try:
                            source_data_item, source_labels_item = utils.get_next_batch(
                                self.dataloaders, self.iters, src[idx], self.device
                            )
                            source_data.append(source_data_item)
                            source_labels.append(source_labels_item)
                        except StopIteration:
                            break

                # forward
                self.optimizer.zero_grad()
                data = torch.cat(source_data + [target_data], dim=0)
                f_shared = self.G_shared(data)
                f = f_shared.chunk(self.num_source + 1, dim=0)

                '''
                # Specific feature extractors defined in the paper will not be used.
                f = [self.Gs_specific[j](f[j]) for j in range(self.num_source)]
                '''

                '''
                # LMMD defined in the paper will not be used.
                f_specific_t = [fs(f_shared_t) for fs in self.Gs_specific]
                y_s = [self.Cs[j](f_specific[j]) for j in range(self.num_source)]
                '''

                # 修复：兼容分类器返回多个值的情况
                y_s = []
                for j in range(self.num_source):
                    c_output = self.Cs[j](f[j])
                    if isinstance(c_output, tuple):
                        y_s.append(c_output[0])  # 第一个输出是分类logits
                    else:
                        y_s.append(c_output)

                loss_cls = 0.0
                for j in range(self.num_source):
                    loss_cls += F.cross_entropy(y_s[j], source_labels[j])
                loss_sum_mmd = 0.0
                for k in range(self.num_source):
                    loss_sum_mmd += self.mkmmd(f[k], f[-1])

                '''
                # LMMD defined in the paper will not be used.
                y_t = [self.Cs[j](f_specific_t[j]) for j in range(self.num_source)]
                y_t = [F.softmax(data, dim=1) for data in y_t] 
                for k in range(self.num_source):
                    oh_label = one_hot(source_labels[k], args.num_classes)
                    for j in range(args.num_classes):
                        w_src = oh_label[:, j].view(-1, 1).to(self.device)
                        w_tgt = y_t[k][:, j].view(-1, 1).to(self.device)
                        loss_sum_mmd += self.mkmmd(w_src*f_specific[k], w_tgt*f_specific_t[k])
                '''

                loss = loss_cls + tradeoff[0] * loss_sum_mmd

                for j in range(self.num_source):
                    epoch_acc[f'Source Data {j}'] += utils.get_accuracy(y_s[j], source_labels[j])

                epoch_loss['Source Classifier'] += loss_cls.item()
                epoch_loss['MMD'] += loss_sum_mmd.item()

                # backward
                loss.backward()
                self.optimizer.step()

            # Print the train and val information via each epoch
            for key in epoch_loss.keys():
                logging.info('Train-Loss {}: {:.4f}'.format(key, epoch_loss[key] / max(num_iter, 1)))
            for key in epoch_acc.keys():
                logging.info('Train-Acc {}: {:.4f}'.format(key, epoch_acc[key] / max(num_iter, 1)))

            # 验证集评估 - 获取Acc和F1
            new_acc, new_f1 = self.test()

            # 更新最优 Accuracy
            if new_acc >= best_acc:
                best_acc = new_acc
                best_acc_epoch = epoch
                best_acc_f1_at_epoch = new_f1
                self.save_model_acc()  # 保存最优Acc模型

            # 更新最优 F1
            if new_f1 >= best_f1:
                best_f1 = new_f1
                best_f1_epoch = epoch
                best_f1_acc_at_epoch = new_acc
                self.save_model_f1()  # 保存最优F1模型

            # 当前epoch的结果
            logging.info(f"Epoch {epoch}: Val-Acc={new_acc:.4f}, Val-F1={new_f1:.4f}")

            # 最佳结果汇总
            logging.info("Best so far → "
                         f"Acc: Epoch {best_acc_epoch} (Acc={best_acc:.4f}, F1={best_acc_f1_at_epoch:.4f}) | "
                         f"F1: Epoch {best_f1_epoch} (Acc={best_f1_acc_at_epoch:.4f}, F1={best_f1:.4f})")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # 训练结束汇总
        logging.info('=' * 70)
        logging.info('Training finished! Best results summary:')
        logging.info(f'Best Accuracy: {best_acc:.4f} (Epoch {best_acc_epoch}, F1={best_acc_f1_at_epoch:.4f})')
        logging.info(f'Best F1-score: {best_f1:.4f} (Epoch {best_f1_epoch}, Acc={best_f1_acc_at_epoch:.4f})')
        logging.info('=' * 70)

    def test(self):
        self.G_shared.eval()
        '''
        # Specific feature extractors defined in the paper will not be used.
        self.Gs_specific.eval()
        '''
        self.Cs.eval()
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

                feat_tgt = self.G_shared(target_data)
                '''
                # Specific feature extractors defined in the paper will not be used.
                feat_tgt = [self.Gs_specific[j](feat_tgt) for j in range(self.num_source)]
                logits_tgt = [self.Cs[j](feat_tgt[j]) for j in range(self.num_source)]
                '''
                # 修复：兼容分类器返回多个值的情况
                logits_tgt = []
                for j in range(self.num_source):
                    c_output = self.Cs[j](feat_tgt)
                    if isinstance(c_output, tuple):
                        logits_tgt.append(c_output[0])  # 第一个输出是分类logits
                    else:
                        logits_tgt.append(c_output)

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