import torch
import logging
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict

import utils
import model_base
from train_utils import InitTrain

#original
# class AutomaticUpdateDomainWeightModule(object):
#     def __init__(self, num_domains: int, eta: float, device):
#         self.domain_weight = torch.ones(num_domains).to(device) / num_domains
#         self.eta = eta
#
#     def get_domain_weight(self, sampled_domain_idxes):
#         """Get domain weight to calculate final objective.
#
#         Inputs:
#             - sampled_domain_idxes (list): sampled domain indexes in current mini-batch
#
#         Shape:
#             - sampled_domain_idxes: :math:`(D, )` where D means the number of sampled domains in current mini-batch
#             - Outputs: :math:`(D, )`
#         """
#         domain_weight = self.domain_weight[sampled_domain_idxes]
#         domain_weight = domain_weight / domain_weight.sum()
#         return domain_weight
#
#     def update(self, sampled_domain_losses: torch.Tensor, sampled_domain_idxes):
#         """Update domain weight using loss of current mini-batch.
#
#         Inputs:
#             - sampled_domain_losses (tensor): loss of among sampled domains in current mini-batch
#             - sampled_domain_idxes (list): sampled domain indexes in current mini-batch
#
#         Shape:
#             - sampled_domain_losses: :math:`(D, )` where D means the number of sampled domains in current mini-batch
#             - sampled_domain_idxes: :math:`(D, )`
#         """
#         sampled_domain_losses = sampled_domain_losses.detach()
#
#         for loss, idx in zip(sampled_domain_losses, sampled_domain_idxes):
#             self.domain_weight[idx] *= (self.eta * loss).exp()
#
#
# class Trainset(InitTrain):
#
#     def __init__(self, args):
#         super(Trainset, self).__init__(args)
#         self.model = model_base.BaseModel(input_size=1, num_classes=args.num_classes,
#                                        dropout=args.dropout).to(self.device)
#         self.domain_weight_module = AutomaticUpdateDomainWeightModule(num_domains=self.num_source,
#                                        eta=1e-2, device=self.device)
#         self._init_data()
#
#     def save_model(self):
#         torch.save({
#             'model': self.model.state_dict()
#             }, self.args.save_path + '.pth')
#         logging.info('Model saved to {}'.format(self.args.save_path + '.pth'))
#
#     def load_model(self):
#         logging.info('Loading model from {}'.format(self.args.load_path))
#         ckpt = torch.load(self.args.load_path)
#         self.model.load_state_dict(ckpt['model'])
#
#     def train(self):
#         args = self.args
#         src = args.source_name
#
#         if args.train_mode != 'multi_source':
#             raise Exception("For this model, invalid train mode: {}".format(args.train_mode))
#
#         self.optimizer = self._get_optimizer(self.model)
#         self.lr_scheduler = self._get_lr_scheduler(self.optimizer)
#
#         best_acc = 0.0
#         best_epoch = 0
#
#         domain_idxes = list(range(self.num_source))
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
#             self.model.train()
#             epoch_loss = defaultdict(float)
#
#             num_iter = len(self.dataloaders['train'])
#             for i in tqdm(range(num_iter), ascii=True):
#                 source_data, source_labels = [], []
#                 for idx in range(self.num_source):
#                     source_data_item, source_labels_item = utils.get_next_batch(self.dataloaders,
#                                                 self.iters, src[idx], self.device)
#                     source_data.append(source_data_item)
#                     source_labels.append(source_labels_item)
#
#                 # forward
#                 cls_acc = 0
#                 self.optimizer.zero_grad()
#                 loss_per_domain = torch.zeros(self.num_source).to(self.device)
#                 for idx in range(self.num_source):
#                     y_per_domain, _ = self.model(source_data[idx])
#                     loss_per_domain[idx] = F.cross_entropy(y_per_domain, source_labels[idx])
#                     cls_acc += utils.get_accuracy(y_per_domain, source_labels[idx]) / self.num_source
#
#                 # update domain weight
#                 self.domain_weight_module.update(loss_per_domain, domain_idxes)
#                 domain_weight = self.domain_weight_module.get_domain_weight(domain_idxes)
#
#                 # weighted cls loss
#                 loss = (loss_per_domain * domain_weight).sum()
#                 epoch_acc['Source Data']  += cls_acc
#
#                 epoch_loss['Source Classifier'] += loss
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
#         self.model.eval()
#         acc = 0.0
#         iters = iter(self.dataloaders['val'])
#         num_iter = len(iters)
#         with torch.no_grad():
#             for i in tqdm(range(num_iter), ascii=True):
#                 target_data, target_labels, _ = next(iters)
#                 target_data, target_labels = target_data.to(self.device), target_labels.to(self.device)
#                 pred = self.model(target_data)
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

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import logging
import utils


class AutomaticUpdateDomainWeightModule(object):
    def __init__(self, num_domains: int, eta: float, device):
        self.domain_weight = torch.ones(num_domains).to(device) / num_domains
        self.eta = eta

    def get_domain_weight(self, sampled_domain_idxes):
        """Get domain weight to calculate final objective."""
        domain_weight = self.domain_weight[sampled_domain_idxes]
        domain_weight = domain_weight / domain_weight.sum()
        return domain_weight

    def update(self, sampled_domain_losses: torch.Tensor, sampled_domain_idxes):
        """Update domain weight using loss of current mini-batch."""
        sampled_domain_losses = sampled_domain_losses.detach()

        for loss, idx in zip(sampled_domain_losses, sampled_domain_idxes):
            self.domain_weight[idx] *= (self.eta * loss).exp()


class Trainset(InitTrain):
    def __init__(self, args):
        super(Trainset, self).__init__(args)
        self.model = model_base.BaseModel(
            input_size=1,
            num_classes=args.num_classes,
            dropout=args.dropout
        ).to(self.device)

        # 根据训练模式决定是否使用域权重模块
        self.train_mode = args.train_mode
        if self.train_mode == 'multi_source':
            self.domain_weight_module = AutomaticUpdateDomainWeightModule(
                num_domains=self.num_source,
                eta=1e-2,
                device=self.device
            )
        else:
            self.domain_weight_module = None

        self._init_data()

        # 自动检测任务类型并设置F1参数
        self.num_classes = args.num_classes
        self.f1_average = self._get_f1_average()
        logging.info(f"Task: {self.num_classes}-class classification | F1 average: '{self.f1_average}'")
        logging.info(f"Training mode: {self.train_mode}")

    def _get_f1_average(self):
        """根据类别数自动选择F1计算方式"""
        if self.num_classes == 2:
            return 'binary'  # 二分类专用
        elif self.num_classes > 2:
            return 'weighted'  # 多分类推荐
        else:
            return 'micro'  # 单分类（理论上不会发生）

    def save_model_acc(self):
        save_path = self.args.save_path + '_best_acc.pth'
        torch.save({'model': self.model.state_dict()}, save_path)
        logging.info('ACC Model saved to {}'.format(save_path))

    def save_model_f1(self):
        save_path = self.args.save_path + '_best_f1.pth'
        torch.save({'model': self.model.state_dict()}, save_path)
        logging.info('F1 Model saved to {}'.format(save_path))

    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path)
        self.model.load_state_dict(ckpt['model'])

    def train(self):
        args = self.args
        src = args.source_name

        # 修改：支持 single_source 和 multi_source
        if args.train_mode not in ['single_source', 'multi_source']:
            raise Exception("For this model, invalid train mode: {}".format(args.train_mode))

        self.optimizer = self._get_optimizer(self.model)
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)

        # 初始化最优指标记录
        best_acc = 0.0
        best_acc_epoch = 0
        best_acc_f1_at_epoch = 0.0

        best_f1 = 0.0
        best_f1_epoch = 0
        best_f1_acc_at_epoch = 0.0

        # 根据训练模式设置域索引
        if self.train_mode == 'multi_source':
            domain_idxes = list(range(self.num_source))
        else:  # single_source
            domain_idxes = [0]  # 只有一个域

        for epoch in range(1, args.max_epoch + 1):
            self.current_epoch = epoch
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-' * 5)

            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))

            epoch_acc = defaultdict(float)
            self.model.train()
            epoch_loss = defaultdict(float)

            num_iter = len(self.dataloaders['train'])
            if num_iter == 0:
                logging.error("Training dataloader is empty!")
                break

            for i in tqdm(range(num_iter), ascii=True):
                # 根据训练模式获取数据
                if self.train_mode == 'multi_source':
                    # 多源模式：获取所有源域的数据
                    source_data, source_labels = [], []
                    try:
                        for idx in range(self.num_source):
                            source_data_item, source_labels_item = utils.get_next_batch(
                                self.dataloaders, self.iters, src[idx], self.device
                            )
                            source_data.append(source_data_item)
                            source_labels.append(source_labels_item)
                    except StopIteration:
                        break
                else:  # single_source
                    # 单源模式：只获取第一个源域的数据
                    try:
                        source_data_item, source_labels_item = utils.get_next_batch(
                            self.dataloaders, self.iters, src[0], self.device
                        )
                        source_data = [source_data_item]
                        source_labels = [source_labels_item]
                    except StopIteration:
                        break

                # forward
                cls_acc = 0
                self.optimizer.zero_grad()

                if self.train_mode == 'multi_source':
                    loss_per_domain = torch.zeros(self.num_source).to(self.device)

                    for idx in range(self.num_source):
                        outputs = self.model(source_data[idx])
                        if isinstance(outputs, tuple):
                            y_per_domain = outputs[0]
                        else:
                            y_per_domain = outputs

                        loss_per_domain[idx] = F.cross_entropy(y_per_domain, source_labels[idx])
                        cls_acc += utils.get_accuracy(y_per_domain, source_labels[idx]) / self.num_source

                    # 更新域权重（仅多源模式）
                    if self.domain_weight_module is not None:
                        self.domain_weight_module.update(loss_per_domain, domain_idxes)
                        domain_weight = self.domain_weight_module.get_domain_weight(domain_idxes)
                        loss = (loss_per_domain * domain_weight).sum()
                    else:
                        loss = loss_per_domain.mean()

                else:  # single_source
                    # 单源模式：直接计算损失
                    outputs = self.model(source_data[0])
                    if isinstance(outputs, tuple):
                        y_per_domain = outputs[0]
                    else:
                        y_per_domain = outputs

                    loss = F.cross_entropy(y_per_domain, source_labels[0])
                    cls_acc = utils.get_accuracy(y_per_domain, source_labels[0])

                epoch_acc['Source Data'] += cls_acc
                epoch_loss['Source Classifier'] += loss.item()

                # backward
                loss.backward()
                self.optimizer.step()

            # 打印训练和验证信息
            for key in epoch_loss.keys():
                logging.info('Train-Loss {}: {:.4f}'.format(key, epoch_loss[key] / max(num_iter, 1)))
            for key in epoch_acc.keys():
                logging.info('Train-Acc {}: {:.4f}'.format(key, epoch_acc[key] / max(num_iter, 1)))

            # 验证集评估
            new_acc, new_f1 = self.test()

            # 更新最优指标
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

        # 训练结束汇总
        logging.info('=' * 70)
        logging.info('Training finished! Best results summary:')
        logging.info(f'Best Accuracy: {best_acc:.4f} (Epoch {best_acc_epoch}, F1={best_acc_f1_at_epoch:.4f})')
        logging.info(f'Best F1-score: {best_f1:.4f} (Epoch {best_f1_epoch}, Acc={best_f1_acc_at_epoch:.4f})')
        logging.info('=' * 70)

    def test(self):
        self.model.eval()
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

                outputs = self.model(target_data)
                if isinstance(outputs, tuple):
                    pred_logits = outputs[0]
                else:
                    pred_logits = outputs

                pred_labels = torch.argmax(pred_logits, dim=1)

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
            logging.warning(f"Only {len(unique_labels)} class(es) present in validation set.")
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