import numpy as np
import os
import torch
import random
import torch.nn as nn
from torch.nn import functional as F
from sklearn.metrics import f1_score


def seed_everything(seed):
    """
    固定各类随机种子，方便消融实验
    """
    random.seed(seed)  
    os.environ['PYTHONHASHSEED'] = str(seed)  
    np.random.seed(seed) 
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = True  


class ModelTrainer(object):

    @staticmethod
    def train(data_loader, model, loss_f, optimizer, epoch_id, device, max_epoch, num_classes):
        model.train()

        conf_mat = np.zeros((num_classes, num_classes))
        loss_sigma = []
        all_labels = []
        all_predictions = []

        for idx, data in enumerate(data_loader):
            inputs, labels = data
            inputs = inputs.to(torch.float32)
            labels = labels.type(torch.LongTensor)

            inputs, labels = inputs.to(device), labels.to(device)

            outputs, _ = model(inputs)

            optimizer.zero_grad()
            loss = loss_f(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.

            # 统计loss值
            loss_sigma.append(loss.item())
            acc_avg = conf_mat.trace() / conf_mat.sum()
            f1_avg = f1_score(all_labels, all_predictions, average="weighted")

            # 每50个iteration 打印一次训练信息, loss为50个iteration的均值
            if idx % 50 == 50 - 1:
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc: {:.2%} F1: {:.2%}".format(
                    epoch_id + 1, max_epoch, idx + 1, len(data_loader), np.mean(loss_sigma), acc_avg, f1_avg
                ))

        return np.mean(loss_sigma), acc_avg, f1_avg

    @staticmethod
    def valid(data_loader, model, loss_f, device, num_classes):
        model.eval()

        conf_mat = np.zeros((num_classes, num_classes))
        loss_sigma = []
        all_labels = []
        all_predictions = []

        for idx, data in enumerate(data_loader):

            inputs, labels = data
            inputs = inputs.to(torch.float32)
            labels = labels.type(torch.LongTensor)

            inputs, labels = inputs.to(device), labels.to(device)

            outputs, _ = model(inputs)
            loss = loss_f(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.

            # 统计loss值t
            loss_sigma.append(loss.item())

        acc_avg = conf_mat.trace() / conf_mat.sum()
        f1_avg = f1_score(all_labels, all_predictions, average="weighted")

        return np.mean(loss_sigma), acc_avg, f1_avg

    @staticmethod
    def train_dls(data_loader, model_e_p, model_e_a, ce_loss, kl_loss1, kl_loss2, optimizer1, optimizer2, epoch_id, device,
                  max_epoch, Beta, r_a, num_classes):
        model_e_p.train()
        model_e_a.train()

        conf_mat = np.zeros((num_classes, num_classes))
        loss_sigma = []
        all_labels = []
        all_predictions = []

        for idx, data in enumerate(data_loader):
            inputs, labels = data
            # 准备数据
            inputs = inputs.to(torch.float32)
            labels = labels.type(torch.LongTensor)
            y_oneHot = ModelTrainer.onehot(labels.shape[0], labels, classes=num_classes)
            inputs, labels, y_oneHot = inputs.to(device), labels.to(device), y_oneHot.to(device)

            # ----------------
            # train model_e_p
            # ----------------

            outputs, extracted_features = model_e_p(inputs)
            adjusted_label = model_e_a(inputs, y_oneHot)

            optimizer1.zero_grad()
            loss_hard = ce_loss(outputs, labels)
            loss_soft = kl_loss1(F.log_softmax(outputs, -1), adjusted_label)
            model_e_p_total_loss = (1 - Beta) * loss_hard + Beta * loss_soft

            model_e_p_total_loss.backward(retain_graph=True)
            optimizer1.step()

            # ----------------
            # train model_e_a
            # ----------------
            adjusted_label = model_e_a(inputs, y_oneHot)
            _, extracted_features = model_e_p(inputs)

            optimizer2.zero_grad()
            y_adj_true = ModelTrainer.area(extracted_features, y_oneHot, Ra=r_a)
            loss_adj = kl_loss2(torch.log(adjusted_label), y_adj_true)
            model_e_a_total_loss = loss_adj + 1 * model_e_p_total_loss.detach()
            model_e_a_total_loss.backward(inputs=list(model_e_a.parameters()))
            optimizer2.step()

            if idx == 1:
                a = adjusted_label.cpu().detach().numpy()
                b = y_oneHot

            # 统计预测值
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.

            # 统计loss值
            loss_sigma.append(model_e_p_total_loss.item())
            acc_avg = conf_mat.trace() / conf_mat.sum()
            f1_avg = f1_score(all_labels, all_predictions, average="weighted")

            # 每50个iteration 打印一次训练信息, loss为50个iteration的均值
            if idx % 50 == 50 - 1:
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc: {:.2%} F1: {:.2%}".format(
                    epoch_id + 1, max_epoch, idx + 1, len(data_loader), np.mean(loss_sigma), acc_avg, f1_avg
                ))

        return np.mean(loss_sigma), acc_avg, f1_avg

    @staticmethod
    def valid_dls(data_loader, model_e_p, model_e_a, loss1, loss2, device, Beta, num_classes):
        model_e_p.eval()
        model_e_a.eval()

        conf_mat = np.zeros((num_classes, num_classes))
        loss_sigma = []
        all_labels = []
        all_predictions = []

        for idx, data in enumerate(data_loader):

            inputs, labels = data

            # 准备数据
            inputs = inputs.to(torch.float32)
            labels = labels.type(torch.LongTensor)
            y_oneHot = ModelTrainer.onehot(labels.shape[0], labels, classes=num_classes)
            inputs, labels, y_oneHot = inputs.to(device), labels.to(device), y_oneHot.to(device)

            outputs, extracted_features = model_e_p(inputs)
            adjusted_label = model_e_a(inputs, y_oneHot)

            loss_hard = loss1(outputs, labels)
            loss_soft = loss2(F.log_softmax(outputs, -1), adjusted_label)
            loss = Beta * loss_hard + (1 - Beta) * loss_soft

            # 统计预测值
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.

            # 统计loss值
            loss_sigma.append(loss.item())

        acc_avg = conf_mat.trace() / conf_mat.sum()
        f1_avg = f1_score(all_labels, all_predictions, average="weighted")

        return np.mean(loss_sigma), acc_avg, f1_avg

    @staticmethod
    def area(X, y, Ra=0.9):
        factor = Ra / (1 - Ra)
        zero = torch.zeros_like(y)
        one = torch.ones_like(y)
        y_inverse = torch.where(y == 0, one, zero)
        X = y_inverse * X
        phi = torch.log(factor * torch.sum(torch.exp(X), dim=1)).unsqueeze(-1)
        out = torch.add(phi * y, X)
        out = F.softmax(out, dim=1)
        return out

    @staticmethod
    def onehot(b, y, classes=52):
        y = y.unsqueeze(-1)
        y = torch.zeros(b, classes).scatter_(1, y, 1)
        return y
