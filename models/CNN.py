import torch
import logging
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict

import utils
import model_base
from train_utils import InitTrain
        
################original
# class Trainset(InitTrain):
#
#     def __init__(self, args):
#         super(Trainset, self).__init__(args)
#         self.model = model_base.BaseModel(input_size=1, num_classes=args.num_classes,
#                                        dropout=args.dropout).to(self.device)
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
#
#         if args.train_mode == 'single_source':
#             src = args.source_name[0]
#         elif args.train_mode == 'source_combine':
#             src = args.source_name
#         elif args.train_mode == 'multi_source':
#             raise Exception("This model cannot be trained in multi_source mode.")
#
#         self.optimizer = self._get_optimizer(self.model)
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
#             self.model.train()
#             epoch_loss = defaultdict(float)
#
#             num_iter = len(self.dataloaders['train'])
#             for i in tqdm(range(num_iter), ascii=True):
#                 source_data, source_labels = utils.get_next_batch(self.dataloaders,
#                                             self.iters, src, self.device)
#                 # forward
#                 self.optimizer.zero_grad()
#                 pred, _ = self.model(source_data)
#                 loss = F.cross_entropy(pred, source_labels)
#                 epoch_acc['Source Data']  += utils.get_accuracy(pred, source_labels)
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

#################acc+f1


import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from collections import defaultdict
import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm


class Trainset(InitTrain):

    def __init__(self, args):
        super(Trainset, self).__init__(args)
        self.model = model_base.BaseModel(
            input_size=1,
            num_classes=args.num_classes,
            dropout=args.dropout
        ).to(self.device)
        self.train_mode = args.train_mode
        self.domain_weight_module = utils.AutomaticUpdateDomainWeightModule(
            num_domains=self.num_source, eta=1e-2, device=self.device
        ) if self.train_mode == 'multi_source' else None
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
        # 保存最优Acc模型（添加后缀区分）
        save_path = self.args.save_path + '_best_acc.pth'
        torch.save({
            'model': self.model.state_dict(),
            'epoch': self.current_epoch if hasattr(self, 'current_epoch') else 0
        }, save_path)
        logging.info('ACC Model saved to {}'.format(save_path))

    def save_model_f1(self):
        # 保存最优F1模型（添加后缀区分）
        save_path = self.args.save_path + '_best_f1.pth'
        torch.save({
            'model': self.model.state_dict(),
            'epoch': self.current_epoch if hasattr(self, 'current_epoch') else 0
        }, save_path)
        logging.info('F1 Model saved to {}'.format(save_path))

    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path)
        self.model.load_state_dict(ckpt['model'])

    def train(self):
        args = self.args

        if args.train_mode == 'single_source':
            src = args.source_name[0]
        elif args.train_mode == 'source_combine':
            src = args.source_name
        elif args.train_mode == 'multi_source':
            src = args.source_name
            domain_idxes = list(range(self.num_source))

        self.optimizer = self._get_optimizer(self.model)
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

            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))

            self.model.train()
            epoch_loss = defaultdict(float)
            epoch_acc = defaultdict(float)
            num_iter = len(self.dataloaders['train'])

            # 检查训练集是否为空
            if num_iter == 0:
                logging.error("Training dataloader is empty!")
                break

            for i in tqdm(range(num_iter), ascii=True):
                self.optimizer.zero_grad()

                if args.train_mode == 'multi_source':
                    loss_per_domain = torch.zeros(self.num_source).to(self.device)
                    for idx in range(self.num_source):
                        try:
                            source_data, source_labels = utils.get_next_batch(
                                self.dataloaders, self.iters, src[idx], self.device
                            )
                        except StopIteration:
                            break
                        outputs = self.model(source_data)
                        pred = outputs[0] if isinstance(outputs, tuple) else outputs
                        loss_per_domain[idx] = F.cross_entropy(pred, source_labels)
                        epoch_acc['Source Data'] += utils.get_accuracy(pred, source_labels) / self.num_source
                    else:
                        self.domain_weight_module.update(loss_per_domain, domain_idxes)
                        domain_weight = self.domain_weight_module.get_domain_weight(domain_idxes)
                        loss = (loss_per_domain * domain_weight).sum()
                        epoch_loss['Source Classifier'] += loss.item()
                        loss.backward()
                        self.optimizer.step()
                    continue
                else:
                    try:
                        source_data, source_labels = utils.get_next_batch(
                            self.dataloaders, self.iters, src, self.device
                        )
                    except StopIteration:
                        break

                    pred, _ = self.model(source_data)
                    loss = F.cross_entropy(pred, source_labels)

                    epoch_acc['Source Data'] += utils.get_accuracy(pred, source_labels)
                    epoch_loss['Source Classifier'] += loss.item()

                    loss.backward()
                    self.optimizer.step()

            # 训练日志
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
        self.model.eval()
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

        # 关键修复：检查验证集是否为空
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

                pred_logits = self.model(target_data)
                pred_labels = torch.argmax(pred_logits, dim=1)

                all_preds.extend(pred_labels.cpu().numpy())
                all_labels.extend(target_labels.cpu().numpy())

        # 关键修复：检查是否有有效预测
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