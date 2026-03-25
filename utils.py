import os
import sys
import json
import math
import torch
from tqdm import tqdm
from model.DiceLoss import DiceLoss, DiceLossV2
from sklearn.metrics import average_precision_score

import torch
from torch import nn
from typing import Optional
from tqdm import tqdm
import sys

def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    type_: Optional[str] = None,
    alpha_start: int = 0,
    alpha_end: int = 100,
    ce_loss: Optional[nn.Module] = None,
    dice_loss: Optional[nn.Module] = None
):
    assert type_ in ['class', 'seg'], "train type is error."
    if ce_loss is None:
        ce_loss = nn.CrossEntropyLoss()
    if dice_loss is None:
        # dice_loss = nn.BCELoss()
        dice_loss = DiceLoss()

    model.train()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    accu_iou = torch.zeros(1).to(device)
    optimizer.zero_grad()
    sample_num = 0

    def get_A_and_B():
        if epoch < alpha_start:
            return 0., 1.
        elif alpha_start <= epoch < alpha_end:
            a = (epoch - alpha_start) / (alpha_end - alpha_start)
            return a, 1 - a
        else:
            return 1., 0.

    cls_weight,seg_weight = get_A_and_B()
    print(f"cls_weight:{cls_weight},seg_weight:{seg_weight}")
    if type_ == 'seg' and seg_weight == 0.:
        return accu_loss.item(), accu_num.item()
    if type_ == 'class' and cls_weight == 0.:
        return accu_loss.item(), accu_num.item()

    data_loader = tqdm(data_loader, file=sys.stdout)
    step = 0
    for step, data in enumerate(data_loader):
        images, labels = data
        labels = labels.to(device)
        batch_size = images.shape[0]
        sample_num += batch_size

        pred_seg, pred_class = model(images.to(device))

        if type_ == 'seg':
            accu_iou += calculate_iou(pred_seg, labels, 1)
            loss_dice = dice_loss(pred_seg, labels)
            loss = loss_dice * seg_weight
        else:
            pred_classes = torch.max(pred_class, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels).sum()
            loss_ce = ce_loss(pred_class, labels)
            loss = loss_ce * cls_weight

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        accu_loss += loss.detach().item()

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        data_loader.desc = "[train epoch {}] loss: {:.3f}, iou: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_iou.item() / sample_num,
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )

        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

    avg_loss = accu_loss.item() / (step + 1) if step >= 0 else float('inf')
    if type_ == 'seg':
        avg_acc = accu_iou.item() / sample_num if sample_num > 0 else float('nan')
    else:
        avg_acc = accu_num.item() / sample_num if sample_num > 0 else float('nan')
    return avg_loss, avg_acc

# def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, type=None):
#     assert type in ['class', 'seg'], "train type is error."
#     model.train()
#     ce_loss = torch.nn.CrossEntropyLoss()
#     dice_loss = torch.nn.BCELoss()  # 分割损失
#     # dice_loss = DiceLoss()  # 分割损失
#     accu_loss = torch.zeros(1).to(device)
#     accu_num = torch.zeros(1).to(device)
#     accu_iou = torch.zeros(1).to(device)
#     optimizer.zero_grad()
#     sample_num = 0
#
#     def get_A_and_B(alpha_start=0, alpha_end=150):
#         # 分割-单任务-start-双任务-end-分类-单任务
#         # alpha_start = 0 alpha_end = 0 纯分类任务
#         # alpha_start = 0 alpha_end = 150 纯双任务
#         # alpha_start = 150 alpha_end = 150 纯分割任务
#         if epoch < alpha_start:
#             return 0, 1
#         elif epoch > alpha_start and epoch < alpha_end:
#             a = (epoch - alpha_start) / (alpha_end - alpha_start)
#             return a, 1 - a
#         else:
#             return 1, 0
#
#     a, b = get_A_and_B()
#     if type == 'seg' and b == 0:
#         return accu_loss, accu_num
#     if type == 'class' and a == 0:
#         return accu_loss, accu_num
#
#     data_loader = tqdm(data_loader, file=sys.stdout)
#     for step, data in enumerate(data_loader):
#
#         images, labels = data
#         labels = labels.to(device)
#         batch_size = images.shape[0]
#         sample_num += batch_size
#
#         pred_seg, pred_class = model(images.to(device))
#         pred_seg = torch.sigmoid(pred_seg)
#         pred_class = torch.softmax(pred_class, dim=1)
#         if type == 'seg':
#             accu_iou += calculate_iou(pred_seg, labels, 1)
#             loss_dice = dice_loss(pred_seg, labels)
#             loss = loss_dice * b
#         else:
#             # 单标签数据集
#             pred_classes = torch.max(pred_class, dim=1)[1]
#             accu_num += torch.eq(pred_classes, labels.to(device)).sum()
#             loss_ce = ce_loss(pred_class, labels)
#             loss = loss_ce * a
#
#         loss.backward()
#         accu_loss += loss.detach()
#
#         if not torch.isfinite(loss):
#             print('WARNING: non-finite loss, ending training ', loss)
#             sys.exit(1)
#
#         data_loader.desc = "[train epoch {}] loss: {:.3f}, iou: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
#             epoch,
#             accu_loss.item() / (step + 1),
#             accu_iou.item() / sample_num,
#             accu_num.item() / sample_num,
#             optimizer.param_groups[0]["lr"]
#         )
#
#         optimizer.step()
#         optimizer.zero_grad()
#         # update lr
#         lr_scheduler.step()
#
#     return accu_loss.item() / (step + 1), accu_num.item() / sample_num

@torch.no_grad()
def evaluate(model, data_loader, device, epoch, type_=None):
    assert type_ in ['class', 'seg'], "evaluate type is error."

    model.eval()
    accu_iou = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    sample_num = 0

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        labels = labels.to(device)
        sample_num += images.shape[0]

        pred_seg, pred_class = model(images.to(device))
        if type_ == 'seg':
            labels = labels.unsqueeze(1)
            accu_iou += calculate_iou(pred_seg, labels, 1)
        else:
            # 单标签数据集
            pred_classes = torch.max(pred_class, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        data_loader.desc = "[valid epoch {}], iou: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_iou.item() / sample_num,
            accu_num.item() / sample_num
        )
    if type_ == 'seg':
        avg_acc = accu_iou.item() / sample_num if sample_num > 0 else float('nan')
    else:
        avg_acc = accu_num.item() / sample_num if sample_num > 0 else float('nan')

    return avg_acc


def train_only_class(model, optimizer, data_loader, device, epoch, lr_scheduler):
    model.train()
    ce_loss = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()
    sample_num = 0


    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        labels = labels.to(device)
        batch_size = images.shape[0]
        sample_num += batch_size

        pred_class = model(images.to(device))

        # 单标签数据集
        pred_classes = torch.max(pred_class, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = ce_loss(pred_class, labels)

        loss.backward()
        accu_loss += loss.detach()

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

@torch.no_grad()
def evaluate_only_class(model, data_loader, device, epoch):
    model.eval()
    accu_num = torch.zeros(1).to(device)
    sample_num = 0


    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        labels = labels.to(device)
        batch_size = images.shape[0]
        sample_num += batch_size

        pred_class = model(images.to(device))

        # 单标签数据集
        pred_classes = torch.max(pred_class, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        data_loader.desc = "[train epoch {}], acc: {:.3f}".format(
            epoch,
            accu_num.item() / sample_num,
        )

    return accu_num.item() / sample_num


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-2):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())

def save_checkpoint(save_path, model, epoch, optimizer=None, lr_scheduler_seg=None, lr_scheduler_class=None):
    folder_path = os.path.dirname(save_path)
    os.makedirs(folder_path, exist_ok=True)
    checkpoint = {
        "net": model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        'lr_schedule_seg': lr_scheduler_seg.state_dict() if lr_scheduler_seg is not None else None,
        'lr_schedule_class': lr_scheduler_class.state_dict() if lr_scheduler_class is not None else None
    }
    try:
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint for epoch {epoch}")
    except Exception as e:
        print(f"Failed to save checkpoint for epoch {epoch}: {e}")


import numpy as np
from sklearn.metrics import confusion_matrix


def show_confusion_matrix(actual_labels, predicted_labels, model):
    print("Actual labels: ")
    print(actual_labels)
    print("Predicted labels: ")
    print(predicted_labels)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of trainable parameters in the model = {count_parameters(model)}")

    # Confusion matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    from matplotlib import pyplot as plt

    cnf_matrix = confusion_matrix(actual_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix)

    disp.plot()
    plt.show()

    # Specificity
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - cnf_matrix.sum(axis=0) - cnf_matrix.sum(axis=1) + np.diag(cnf_matrix)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    TNR = TN / (TN + FP)

    print(f"Class wise specificity:")
    print(f"Specificity = {TNR}\n")

    print(f"Average specificity:")
    print(f"Specificity = {np.average(np.array(TNR))}\n")

    # Accuracy, Sensitivity, Precision, F1 score
    from sklearn.metrics import classification_report

    target_names = ['0', '1', '2', '3', '4', '5', '6', '7']
    print(classification_report(actual_labels, predicted_labels,digits=4))

    # # ROC curve
    # from sklearn import metrics
    #
    # fpr, tpr, thresholds = metrics.roc_curve(actual_labels, predicted_labels)
    # roc_auc = metrics.auc(fpr, tpr)
    # display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    # display.plot()
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc="lower right")
    # plt.show()
    # plt.savefig("test_result.jpg")

    # AUC
    # print(f"AUC = {roc_auc}")


# def calculate_iou(preds, targets, num_classes=2, threshold=0.5):
#     """Calculate total IoU for segmentation task.
#
#     Args:
#         preds (torch.Tensor): Predicted segmentation map of shape [batch_size, 1, height, width].
#         targets (torch.Tensor): Ground truth segmentation map of shape [batch_size, 1, height, width].
#         num_classes (int): Number of classes in the segmentation task.
#         threshold (float): Threshold to convert predictions to binary.
#
#     Returns:
#         float: Total IoU across all samples.
#     """
#     total_iou = 0.0
#     preds = (preds >= threshold).float()
#     preds = preds.squeeze(1).cpu().detach().numpy()  # [batch_size, height, width]
#     targets = targets.squeeze(1).cpu().detach().numpy()  # [batch_size, height, width]
#
#     batch_size = preds.shape[0]
#
#     for i in range(batch_size):
#         sample_ious = []
#         for cls in range(num_classes):
#             pred_inds = preds[i] == 1
#             target_inds = targets[i] == 1
#             intersection = (pred_inds & target_inds).sum()
#             union = (pred_inds | target_inds).sum()
#
#             if union == 0:
#                 sample_ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
#             else:
#                 sample_ious.append(intersection / union)
#
#         sample_iou = np.nanmean(sample_ious)  # Ignore NaN values when calculating mean
#         if not np.isnan(sample_iou):
#             total_iou += sample_iou
#
#     return total_iou

def calculate_metrics(preds, targets, threshold=0.5):
    """
    Calculate IoU and Dice coefficient for binary segmentation task.

    Args:
        preds (torch.Tensor): Predicted segmentation map of shape [batch_size, 1, height, width].
        targets (torch.Tensor): Ground truth segmentation map of shape [batch_size, 1, height, width].
        threshold (float): Threshold to convert predictions to binary.

    Returns:
        tuple: (iou, dice) both are floats representing the metrics.
    """
    # Convert predictions to binary and move to CPU
    preds = (preds >= threshold).float().squeeze(1).cpu().detach().numpy()
    targets = targets.squeeze(1).cpu().numpy()

    # Ensure the inputs are boolean arrays
    preds = preds.astype(bool)
    targets = targets.astype(bool)

    # Calculate intersection and union
    intersection = (preds & targets).sum(axis=(1, 2))
    union = (preds | targets).sum(axis=(1, 2))

    # Calculate IoU
    iou = (intersection / union).mean()

    # Calculate Dice coefficient
    dice = (2 * intersection / (preds.sum(axis=(1, 2)) + targets.sum(axis=(1, 2)))).mean()

    return float(iou), float(dice)


