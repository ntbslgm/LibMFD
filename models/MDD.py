'''
Paper: Zhang, Y., Liu, T., Long, M. and Jordan, M., 2019, May. Bridging theory and 
    algorithm for domain adaptation. In International conference on machine learning (pp. 7404-7413). PMLR.
Reference code: https://github.com/thuml/Transfer-Learning-Library
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


# def shift_log(x: torch.Tensor, offset: float = 1e-6) -> torch.Tensor:
#
#     return torch.log(torch.clamp(x + offset, max=1.))
#
#
# class MarginDisparityDiscrepancy(nn.Module):
#
#     def __init__(self, source_disparity, target_disparity,
#                  margin: float = 4, reduction: str = 'mean'):
#         super(MarginDisparityDiscrepancy, self).__init__()
#         self.margin = margin
#         self.reduction = reduction
#         self.source_disparity = source_disparity
#         self.target_disparity = target_disparity
#
#     def forward(self, y_s: torch.Tensor, y_s_adv: torch.Tensor, y_t: torch.Tensor, y_t_adv: torch.Tensor,
#                 w_s: torch.Tensor = None, w_t: torch.Tensor = None) -> torch.Tensor:
#
#         source_loss = -self.margin * self.source_disparity(y_s, y_s_adv)
#         target_loss = self.target_disparity(y_t, y_t_adv)
#         if w_s is None:
#             w_s = torch.ones_like(source_loss)
#         source_loss = source_loss * w_s
#         if w_t is None:
#             w_t = torch.ones_like(target_loss)
#         target_loss = target_loss * w_t
#
#         loss = source_loss + target_loss
#         if self.reduction == 'mean':
#             loss = loss.mean()
#         elif self.reduction == 'sum':
#             loss = loss.sum()
#         return loss
#
#
# class ClassificationMarginDisparityDiscrepancy(MarginDisparityDiscrepancy):
#
#     def __init__(self, margin: float = 4, **kwargs):
#         def source_discrepancy(y: torch.Tensor, y_adv: torch.Tensor):
#             _, prediction = y.max(dim=1)
#             return F.cross_entropy(y_adv, prediction, reduction='none')
#
#         def target_discrepancy(y: torch.Tensor, y_adv: torch.Tensor):
#             _, prediction = y.max(dim=1)
#             return -F.nll_loss(shift_log(1. - F.softmax(y_adv, dim=1)), prediction, reduction='none')
#
#         super(ClassificationMarginDisparityDiscrepancy, self).__init__(source_discrepancy, target_discrepancy, margin,
#                                                                        **kwargs)
#
#
# class GeneralModule(nn.Module):
#
#     def __init__(self, args, grl):
#         super(GeneralModule, self).__init__()
#         output_size = 2560
#         self.G = model_base.FeatureExtractor(in_channel=1)
#         self.C1 = model_base.ClassifierMLP(input_size=output_size, output_size=args.num_classes,
#                         dropout=args.dropout, last=None)
#         self.C2 = model_base.ClassifierMLP(input_size=output_size, output_size=args.num_classes,
#                         dropout=args.dropout, last=None)
#         self.grl_layer = utils.WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000,
#                                                        auto_step=False) if grl is None else grl
#
#     def forward(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
#         """"""
#         features = self.G(x)
#         outputs = self.C1(features)
#         features_adv = self.grl_layer(features)
#         outputs_adv = self.C2(features_adv)
#         if self.training:
#             return outputs, outputs_adv
#         else:
#             return outputs
#
#     def step(self):
#         """
#         Gradually increase :math:`\lambda` in GRL layer.
#         """
#         self.grl_layer.step()
#
#
# class Trainset(InitTrain):
#
#     def __init__(self, args):
#         super(Trainset, self).__init__(args)
#         self.mdd = ClassificationMarginDisparityDiscrepancy().to(self.device)
#         grl = utils.GradientReverseLayer()
#         self.model = GeneralModule(args, grl=grl).to(self.device)
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
#             tradeoff = self._get_tradeoff(args.tradeoff, epoch)
#
#             num_iter = len(self.dataloaders['train'])
#             for i in tqdm(range(num_iter), ascii=True):
#                 target_data, target_labels = utils.get_next_batch(self.dataloaders,
#                 						 self.iters, 'train', self.device)
#                 source_data, source_labels = utils.get_next_batch(self.dataloaders,
#             						     self.iters, src, self.device)
#                 # forward
#                 self.optimizer.zero_grad()
#                 data = torch.cat((source_data, target_data), dim=0)
#
#                 outputs, outputs_adv = self.model(data)
#                 y_s, y_t = outputs.chunk(2, dim=0)
#                 y_s_adv, y_t_adv = outputs_adv.chunk(2, dim=0)
#
#                 # compute cross entropy loss on source domain
#                 cls_loss = F.cross_entropy(y_s, source_labels)
#                 # compute margin disparity discrepancy between domains
#                 # for adversarial classifier, minimize negative mdd is equal to maximize mdd
#                 transfer_loss = -self.mdd(y_s, y_s_adv, y_t, y_t_adv)
#                 loss = cls_loss + tradeoff[0] * transfer_loss
#                 self.model.step()
#
#                 epoch_acc['Source train']  += utils.get_accuracy(y_s, source_labels)
#                 epoch_loss['Source domain'] += cls_loss
#                 epoch_loss['MDD'] += transfer_loss
#                 loss.backward()
#                 self.optimizer.step()
#
#             # Print the train and val information via each epoch
#             for key in epoch_loss.keys():
#                 logging.info('Train-Loss {}: {:.4f}'.format(key, epoch_loss[key]/num_iter))
#             for key in epoch_acc.keys():
#                 logging.info('Train-Acc {}: {:.4f}'.format(key, epoch_acc[key]/num_iter))
#
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


def shift_log(x: torch.Tensor, offset: float = 1e-6) -> torch.Tensor:
    return torch.log(torch.clamp(x + offset, max=1.))


class MarginDisparityDiscrepancy(nn.Module):

    def __init__(self, source_disparity, target_disparity,
                 margin: float = 4, reduction: str = 'mean'):
        super(MarginDisparityDiscrepancy, self).__init__()
        self.margin = margin
        self.reduction = reduction
        self.source_disparity = source_disparity
        self.target_disparity = target_disparity

    def forward(self, y_s: torch.Tensor, y_s_adv: torch.Tensor, y_t: torch.Tensor, y_t_adv: torch.Tensor,
                w_s: torch.Tensor = None, w_t: torch.Tensor = None) -> torch.Tensor:

        source_loss = -self.margin * self.source_disparity(y_s, y_s_adv)
        target_loss = self.target_disparity(y_t, y_t_adv)
        if w_s is None:
            w_s = torch.ones_like(source_loss)
        source_loss = source_loss * w_s
        if w_t is None:
            w_t = torch.ones_like(target_loss)
        target_loss = target_loss * w_t

        loss = source_loss + target_loss
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class ClassificationMarginDisparityDiscrepancy(MarginDisparityDiscrepancy):

    def __init__(self, margin: float = 4, **kwargs):
        def source_discrepancy(y: torch.Tensor, y_adv: torch.Tensor):
            _, prediction = y.max(dim=1)
            return F.cross_entropy(y_adv, prediction, reduction='none')

        def target_discrepancy(y: torch.Tensor, y_adv: torch.Tensor):
            _, prediction = y.max(dim=1)
            return -F.nll_loss(shift_log(1. - F.softmax(y_adv, dim=1)), prediction, reduction='none')

        super(ClassificationMarginDisparityDiscrepancy, self).__init__(
            source_discrepancy, target_discrepancy, margin, **kwargs)


class GeneralModule(nn.Module):

    def __init__(self, args, grl):
        super(GeneralModule, self).__init__()
        output_size = 2560
        self.G = model_base.FeatureExtractor(in_channel=1)
        self.C1 = model_base.ClassifierMLP(
            input_size=output_size,
            output_size=args.num_classes,
            dropout=args.dropout,
            last=None
        )
        self.C2 = model_base.ClassifierMLP(
            input_size=output_size,
            output_size=args.num_classes,
            dropout=args.dropout,
            last=None
        )
        self.grl_layer = utils.WarmStartGradientReverseLayer(
            alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False
        ) if grl is None else grl

    def forward(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        features = self.G(x)

        # 修复：兼容分类器返回多个值的情况
        c1_outputs = self.C1(features)
        if isinstance(c1_outputs, tuple):
            outputs = c1_outputs[0]  # 第一个输出是分类logits
        else:
            outputs = c1_outputs

        features_adv = self.grl_layer(features)
        c2_outputs = self.C2(features_adv)
        if isinstance(c2_outputs, tuple):
            outputs_adv = c2_outputs[0]  # 第一个输出是分类logits
        else:
            outputs_adv = c2_outputs

        if self.training:
            return outputs, outputs_adv
        else:
            return outputs

    def step(self):
        """
        Gradually increase :math:`\lambda` in GRL layer.
        """
        self.grl_layer.step()


class Trainset(InitTrain):

    def __init__(self, args):
        super(Trainset, self).__init__(args)
        self.mdd = ClassificationMarginDisparityDiscrepancy().to(self.device)

        # 修复：使用 WarmStartGradientReverseLayer 而不是 GradientReverseLayer
        grl = utils.WarmStartGradientReverseLayer(
            alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False
        )
        self.model = GeneralModule(args, grl=grl).to(self.device)
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
        # 保存最优Acc模型
        save_path = self.args.save_path + '_best_acc.pth'
        torch.save({
            'model': self.model.state_dict()
        }, save_path)
        logging.info('ACC Model saved to {}'.format(save_path))

    def save_model_f1(self):
        # 保存最优F1模型
        save_path = self.args.save_path + '_best_f1.pth'
        torch.save({
            'model': self.model.state_dict()
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

            # Update the learning rate
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))

            # Each epoch has a training and val phase
            epoch_acc = defaultdict(float)

            # Set model to train mode or evaluate mode
            self.model.train()
            epoch_loss = defaultdict(float)
            tradeoff = self._get_tradeoff(args.tradeoff, epoch)

            num_iter = len(self.dataloaders['train'])
            if num_iter == 0:
                logging.error("Training dataloader is empty!")
                break

            for i in tqdm(range(num_iter), ascii=True):
                self.optimizer.zero_grad()

                if args.train_mode == 'multi_source':
                    try:
                        target_data, target_labels = utils.get_next_batch(
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

                    loss_per_domain = torch.zeros(self.num_source).to(self.device)
                    loss_adapt_per_domain = torch.zeros(self.num_source).to(self.device)
                    for idx in range(self.num_source):
                        data = torch.cat((source_data[idx], target_data), dim=0)
                        outputs = self.model(data)
                        if isinstance(outputs, tuple):
                            outputs, outputs_adv = outputs[0], outputs[1]
                        else:
                            outputs, outputs_adv = outputs, None

                        y_s, y_t = outputs.chunk(2, dim=0)
                        y_s_adv, y_t_adv = outputs_adv.chunk(2, dim=0)

                        cls_loss = F.cross_entropy(y_s, source_labels[idx])
                        transfer_loss = -self.mdd(y_s, y_s_adv, y_t, y_t_adv)

                        loss_per_domain[idx] = cls_loss
                        loss_adapt_per_domain[idx] = transfer_loss
                        epoch_acc['Source train'] += utils.get_accuracy(y_s, source_labels[idx]) / self.num_source

                    self.domain_weight_module.update(loss_per_domain, domain_idxes)
                    domain_weight = self.domain_weight_module.get_domain_weight(domain_idxes)
                    loss = (domain_weight * (loss_per_domain + tradeoff[0] * loss_adapt_per_domain)).sum()
                    self.model.step()

                    epoch_loss['Source domain'] += (domain_weight * loss_per_domain).sum().item()
                    epoch_loss['MDD'] += (domain_weight * loss_adapt_per_domain).sum().item()

                    loss.backward()
                    self.optimizer.step()
                    continue

                try:
                    target_data, target_labels = utils.get_next_batch(
                        self.dataloaders, self.iters, 'train', self.device
                    )
                    source_data, source_labels = utils.get_next_batch(
                        self.dataloaders, self.iters, src, self.device
                    )
                except StopIteration:
                    break

                # forward
                data = torch.cat((source_data, target_data), dim=0)

                # 修复：兼容模型返回多个值的情况
                outputs = self.model(data)
                if isinstance(outputs, tuple):
                    outputs, outputs_adv = outputs[0], outputs[1]
                else:
                    # 如果模型只返回一个值，那么 outputs_adv 应该为 None
                    # 但根据 GeneralModule 的设计，训练时应该返回两个值
                    # 这里为了安全起见，我们创建一个虚拟的 outputs_adv
                    outputs_adv = None
                    logging.warning("Model returned single value during training, expected two values.")

                y_s, y_t = outputs.chunk(2, dim=0)
                if outputs_adv is not None:
                    y_s_adv, y_t_adv = outputs_adv.chunk(2, dim=0)
                else:
                    # 如果 outputs_adv 为 None，创建零张量以避免错误
                    y_s_adv = torch.zeros_like(y_s)
                    y_t_adv = torch.zeros_like(y_t)

                # compute cross entropy loss on source domain
                cls_loss = F.cross_entropy(y_s, source_labels)
                # compute margin disparity discrepancy between domains
                # for adversarial classifier, minimize negative mdd is equal to maximize mdd
                transfer_loss = -self.mdd(y_s, y_s_adv, y_t, y_t_adv)
                loss = cls_loss + tradeoff[0] * transfer_loss
                self.model.step()

                epoch_acc['Source train'] += utils.get_accuracy(y_s, source_labels)
                epoch_loss['Source domain'] += cls_loss.item()
                epoch_loss['MDD'] += transfer_loss.item()

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

                # 修复：兼容模型返回多个值的情况
                outputs = self.model(target_data)
                if isinstance(outputs, tuple):
                    pred = outputs[0]  # 取第一个作为分类logits
                else:
                    pred = outputs

                pred_labels = torch.argmax(pred, dim=1)

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