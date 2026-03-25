# utils/metrics.py
import numpy as np
from collections import OrderedDict, defaultdict
import torch

from scipy.stats import hmean, gmean
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import f1_score, confusion_matrix



@torch.no_grad()
def binary_iou(preds, targets, threshold=0.5, eps=1e-6):
    """
    preds: [B, 1, H, W] (logits or probs)
    targets: [B, H, W] or [B, 1, H, W]
    """
    # 确保预测值是概率值（应用sigmoid处理logits）
    if preds.min() < 0 or preds.max() > 1:
        preds = torch.sigmoid(preds)  # 将logits转换为概率

    # 处理目标张量的维度
    if targets.dim() == 3:  # [B, H, W]
        # 目标已经是正确的形式，不需要改变
        targets = targets.float()
    elif targets.dim() == 4 and targets.shape[1] == 1:  # [B, 1, H, W]
        targets = targets.squeeze(1).float()  # 转换为 [B, H, W]

    # 处理预测张量的维度
    if preds.dim() == 4 and preds.shape[1] == 1:  # [B, 1, H, W]
        preds = preds.squeeze(1).float()  # 转换为 [B, H, W]

    # 应用阈值进行二值化
    preds = (preds > threshold).float()
    targets = (targets > 0).float()  # 确保目标也是二值化的

    # 现在 preds 和 targets 都是 [B, H, W]
    intersection = (preds * targets).sum(dim=(1, 2))
    union = (preds + targets - preds * targets).sum(dim=(1, 2))

    # 防止除以零的情况
    union = torch.clamp(union, min=eps)

    iou = (intersection + eps) / (union + eps)

    # 确保IOU值在合理范围内
    iou = torch.clamp(iou, min=0.0, max=1.0)

    return iou.mean()


@torch.no_grad()
def binary_dice(preds, targets, threshold=0.5, eps=1e-6):
    preds = (preds > threshold).float()
    targets = targets.float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (
        preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps
    )
    return dice.mean()





def compute_classification_metrics(y_pred, y_true, num_classes=None):
    """
    计算详细的分类指标，基于参考的Evaluator类进行优化
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    # 基础指标
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)

    # 计算每类准确率
    cm = confusion_matrix(y_true, y_pred)
    if num_classes is not None:
        # 确保混淆矩阵维度正确
        if cm.shape[0] < num_classes or cm.shape[1] < num_classes:
            full_cm = np.zeros((num_classes, num_classes), dtype=cm.dtype)
            full_cm[:cm.shape[0], :cm.shape[1]] = cm
            cm = full_cm

    per_class_acc = np.diag(cm) / (cm.sum(axis=1) + 1e-8)

    # 计算各种平均指标
    mean_per_class_acc = np.mean(per_class_acc)
    worst_case_acc = np.min(per_class_acc) if len(per_class_acc) > 0 else 0.0

    # 计算调和平均和几何平均
    per_class_acc_safe = np.maximum(per_class_acc, 1e-8)  # 避免除零
    hmean_acc = hmean(per_class_acc_safe) if len(per_class_acc) > 0 else 0.0
    gmean_acc = gmean(per_class_acc_safe) if len(per_class_acc) > 0 else 0.0

    # 微平均F1（等同于准确率）
    micro_avg_f1 = 2 * np.sum(cm.diagonal()) / np.sum(cm) if cm.size > 0 else 0.0

    metrics = OrderedDict([
        ('acc', acc),
        ('f1_macro', macro_f1),
        ('f1_micro', micro_f1),
        ('f1_micro_avg', micro_avg_f1),
        ('precision_macro', precision_macro),
        ('recall_macro', recall_macro),
        ('per_class_acc', per_class_acc.tolist()),
        ('mean_per_class_acc', mean_per_class_acc),
        ('worst_case_acc', worst_case_acc),
        ('hmean_acc', hmean_acc),
        ('gmean_acc', gmean_acc),
        ('total_samples', len(y_true)),
        ('correct_predictions', int(np.sum(y_pred == y_true)))
    ])

    return metrics


def print_classification_results(metrics, dataset_name=None):
    """
    打印分类评估结果
    """
    print("\n" + "=" * 60)
    if dataset_name:
        print(f"Classification Results - Dataset: {dataset_name}")
    else:
        print("Classification Results")
    print("=" * 60)

    print(f"Total Samples: {metrics['total_samples']:,}")
    print(f"Correct Predictions: {metrics['correct_predictions']:,}")
    print(f"Overall Accuracy: {metrics['acc']:.2f}%")
    print(f"Error Rate: {100.0 - metrics['acc']:.2f}%")

    print(f"\nMacro F1 Score: {metrics['f1_macro']:.2f}%")
    print(f"Micro F1 Score: {metrics['f1_micro']:.2f}%")
    print(f"Micro Avg F1 Score: {metrics['f1_micro_avg']:.2f}%")
    print(f"Macro Precision: {metrics['precision_macro']:.2f}%")
    print(f"Macro Recall: {metrics['recall_macro']:.2f}%")

    print(f"\nMean Per-Class Accuracy: {metrics['mean_per_class_acc']:.2f}%")
    print(f"Worst Case Accuracy: {metrics['worst_case_acc']:.2f}%")
    print(f"Harmonic Mean Accuracy: {metrics['hmean_acc']:.2f}%")
    print(f"Geometric Mean Accuracy: {metrics['gmean_acc']:.2f}%")

    print(f"\nPer-Class Accuracies:")
    per_class_acc = np.array(metrics['per_class_acc'])
    accs_string = np.array2string(per_class_acc, precision=2)
    print(f"{accs_string}")

    print("=" * 60)


def compute_segmentation_metrics(pred_masks, targets, threshold=0.5):
    """
    计算分割指标
    """
    iou = binary_iou(pred_masks, targets, threshold)
    dice = binary_dice(pred_masks, targets, threshold)

    metrics = {
        'iou': iou.item(),
        'dice': dice.item()
    }

    return metrics


def print_segmentation_results(metrics, dataset_name=None):
    """
    打印分割评估结果
    """
    print("\n" + "=" * 60)
    if dataset_name:
        print(f"Segmentation Results - Dataset: {dataset_name}")
    else:
        print("Segmentation Results")
    print("=" * 60)

    print(f"IOU: {metrics['iou']:.4f}")
    print(f"Dice Coefficient: {metrics['dice']:.4f}")

    print("=" * 60)
