import argparse
import os
import sys

import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, matthews_corrcoef, confusion_matrix)
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset.dataset import DataSetMask
from dataset.ISIC2018 import (DataSetMutliTaskSegment, DataSetMutliTaskClassify,
                              read_train_data_isic2018_class, read_train_data_isic2018_seg,
                              read_test_data_isic2018_class, read_test_data_isic2018_seg)
from dataset.Kvasir import read_train_test_data_kvasir_class, read_data_kvasir_seg
from dataset.covid19 import read_train_test_data_covid19_class, COVID19ClassDataset

from utils import show_confusion_matrix
from analysis.GradCAM import (Init_Setting_DFFV1_Small, Init_Setting_DFFV1_Tiny,
                              Init_Setting_HiFuse_Small, Init_Setting_DFFV1_Base,
                              Init_Setting_DFFV3_Base, Init_Setting_ResNet18, Init_Setting_MIXER,
                              Init_Setting_DEIT, Init_Setting_VIT_32, Init_Setting_VGG19,
                              Init_Setting_VIT_16, Init_Setting_SWIN_B, Init_Setting_FOCAL_B,
                              Init_Setting_CONVNEXT_B, Init_Setting_DFFV1_Mini)


# ==============================================================================
# 分类评估
# ==============================================================================

def class_wise_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    accuracy_per_class = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        total = np.sum(cm[i])
        if total == 0:
            acc = 0.0
        else:
            acc = (tp + tn) / total
        accuracy_per_class.append(acc)
    return accuracy_per_class


def evaluate_metrics(labels_list, pred_classes_list, pred_probs_list=None, num_classes=None):
    acc = accuracy_score(labels_list, pred_classes_list)
    prec = precision_score(labels_list, pred_classes_list, average='macro', zero_division=0)
    rec = recall_score(labels_list, pred_classes_list, average='macro', zero_division=0)
    f1 = f1_score(labels_list, pred_classes_list, average='macro')
    mcc = matthews_corrcoef(labels_list, pred_classes_list)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"MCC: {mcc:.4f}")

    auc = None
    if pred_probs_list is not None and num_classes is not None:
        try:
            auc = roc_auc_score(labels_list, pred_probs_list, average='macro', multi_class='ovr')
            print(f"AUC (Macro): {auc:.4f}")
        except ValueError as e:
            print(f"Could not calculate AUC: {e}")

    return acc, prec, rec, f1, mcc, auc


# ==============================================================================
# 分割评估 (IoU, Dice, HD95, ASSD, Precision, Recall)
# ==============================================================================

def compute_seg_metrics(pred_mask, gt_mask, threshold=0.5, spacing=None):
    """
    计算分割指标。
    pred_mask:  sigmoid 后的概率图或二值 mask，shape (1, H, W) 或 (H, W)
    gt_mask:    ground truth mask，shape (H, W)，值 ∈ {0, 1}
    spacing:    像素间距，用于 HD95/ASSD。默认 (1.0, 1.0)
    返回 dict: IoU, Dice, HD95, ASSD, Precision, Recall
    """
    # 统一处理 shape
    if pred_mask.ndim == 3:
        pred_mask = pred_mask.squeeze(0)

    # 二值化预测
    pred_bin = (pred_mask > threshold).astype(np.uint8)
    gt_bin = gt_mask.astype(np.uint8)

    # ---- IoU / Dice / Precision / Recall (pixel-level) ----
    intersection = (pred_bin & gt_bin).sum()
    union = (pred_bin | gt_bin).sum()
    pred_sum = pred_bin.sum()
    gt_sum = gt_bin.sum()

    iou = float(intersection) / float(union) if union > 0 else 0.0
    dice = 2.0 * float(intersection) / float(pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0
    pixel_precision = float(intersection) / float(pred_sum) if pred_sum > 0 else 0.0
    pixel_recall = float(intersection) / float(gt_sum) if gt_sum > 0 else 0.0

    # ---- HD95 / ASSD (基于轮廓距离) ----
    hd95 = None
    assd = None
    try:
        hd95 = compute_hd95(pred_bin, gt_bin, spacing=spacing)
    except Exception:
        pass
    try:
        assd = compute_assd(pred_bin, gt_bin, spacing=spacing)
    except Exception:
        pass

    return {
        "IoU": iou,
        "Dice": dice,
        "HD95": hd95,
        "ASSD": assd,
        "Precision": pixel_precision,
        "Recall": pixel_recall,
    }


def compute_hd95(pred_bin, gt_bin, spacing=None):
    """计算 95% Hausdorff Distance，基于 scipy 距离变换。"""
    from scipy.ndimage import distance_transform_edt

    if spacing is None:
        spacing = (1.0, 1.0)

    # 提取轮廓：前景 & 背景交界的像素
    pred_border = pred_bin ^ _erode(pred_bin)
    gt_border = gt_bin ^ _erode(gt_bin)

    if pred_border.sum() == 0 or gt_border.sum() == 0:
        return float('inf')

    # 距离变换（带 spacing）
    dt_pred = distance_transform_edt(1 - pred_border, sampling=spacing)
    dt_gt = distance_transform_edt(1 - gt_border, sampling=spacing)

    # 从 pred 轮廓到 gt 轮廓的距离
    dist_p2g = dt_gt[pred_border > 0]
    dist_g2p = dt_pred[gt_border > 0]

    distances = np.concatenate([dist_p2g, dist_g2p])
    return float(np.percentile(distances, 95))


def compute_assd(pred_bin, gt_bin, spacing=None):
    """计算 Average Symmetric Surface Distance。"""
    from scipy.ndimage import distance_transform_edt

    if spacing is None:
        spacing = (1.0, 1.0)

    pred_border = pred_bin ^ _erode(pred_bin)
    gt_border = gt_bin ^ _erode(gt_bin)

    if pred_border.sum() == 0 or gt_border.sum() == 0:
        return float('inf')

    dt_pred = distance_transform_edt(1 - pred_border, sampling=spacing)
    dt_gt = distance_transform_edt(1 - gt_border, sampling=spacing)

    dist_p2g = dt_gt[pred_border > 0]
    dist_g2p = dt_pred[gt_border > 0]

    return float(np.mean(dist_p2g) + np.mean(dist_g2p)) / 2.0


def _erode(bin_mask, kernel_size=3):
    """3×3 最小值滤波 = 二值腐蚀"""
    from scipy.ndimage import minimum_filter
    return minimum_filter(bin_mask, size=kernel_size)


def evaluate_segmentation(model, seg_loader, device, threshold=0.5, spacing=None):
    """
    对分割数据集进行评估，返回各指标的均值。
    model 的 forward 必须返回 (seg_output, class_output)。
    """
    metrics_sum = {"IoU": 0.0, "Dice": 0.0, "HD95": [], "ASSD": [],
                   "Precision": 0.0, "Recall": 0.0}
    n_valid = 0

    with torch.no_grad():
        loader = tqdm(seg_loader, file=sys.stdout, desc="Segmentation eval")
        for images, masks in loader:
            images = images.to(device)
            outputs = model(images)

            if isinstance(outputs, tuple):
                seg_output = outputs[0]  # (B, 1, H, W) — 模型 forward 已做 sigmoid
            elif isinstance(outputs, dict):
                seg_output = outputs.get("seg_mask", list(outputs.values())[0])
            else:
                seg_output = outputs

            # 注意：DFF 模型的 seg_output 已经过 sigmoid（见 dff.py L332）
            seg_probs = seg_output.cpu().numpy()  # (B, 1, H, W)
            masks_np = masks.cpu().numpy()        # (B, 1, H, W)

            # 跳过无前景 mask 的样本
            valid_indices = [i for i in range(seg_probs.shape[0])
                             if masks_np[i].sum() > 0]
            seg_probs = seg_probs[valid_indices]
            masks_np = masks_np[valid_indices]

            for i in range(seg_probs.shape[0]):
                m = compute_seg_metrics(seg_probs[i], masks_np[i].squeeze(0),
                                        threshold=threshold, spacing=spacing)
                metrics_sum["IoU"] += m["IoU"]
                metrics_sum["Dice"] += m["Dice"]
                if m["HD95"] is not None and m["HD95"] != float('inf'):
                    metrics_sum["HD95"].append(m["HD95"])
                if m["ASSD"] is not None and m["ASSD"] != float('inf'):
                    metrics_sum["ASSD"].append(m["ASSD"])
                metrics_sum["Precision"] += m["Precision"]
                metrics_sum["Recall"] += m["Recall"]
                n_valid += 1

    print(f"\n--- Segmentation Results (n={n_valid}) ---")
    print(f"IoU:       {metrics_sum['IoU'] / n_valid:.4f}")
    print(f"Dice:      {metrics_sum['Dice'] / n_valid:.4f}")
    hd95_mean = np.mean(metrics_sum["HD95"]) if metrics_sum["HD95"] else float('inf')
    assd_mean = np.mean(metrics_sum["ASSD"]) if metrics_sum["ASSD"] else float('inf')
    print(f"HD95:      {hd95_mean:.4f}")
    print(f"ASSD:      {assd_mean:.4f}")
    print(f"Precision: {metrics_sum['Precision'] / n_valid:.4f}")
    print(f"Recall:    {metrics_sum['Recall'] / n_valid:.4f}")

    return {
        "IoU": metrics_sum["IoU"] / n_valid if n_valid else 0,
        "Dice": metrics_sum["Dice"] / n_valid if n_valid else 0,
        "HD95": hd95_mean,
        "ASSD": assd_mean,
        "Precision": metrics_sum["Precision"] / n_valid if n_valid else 0,
        "Recall": metrics_sum["Recall"] / n_valid if n_valid else 0,
    }


# ==============================================================================
# 主函数
# ==============================================================================

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    # ---- 数据集配置 ----
    if args.dataset == "isic2018":
        val_data_path = "/ai/data/ISIC2018"
        test_class_data = read_test_data_isic2018_class(val_data_path)
        test_seg_data = read_test_data_isic2018_seg(val_data_path)
        num_classes = 7
    elif args.dataset == "kvasir":
        val_data_path = "/ai/data/Kvasir"
        train_class_data, test_class_data = read_train_test_data_kvasir_class(val_data_path)
        # Kvasir seg: train/val split, 取 test 部分 (test_ratio=0.1 时前10%是test)
        _, test_seg_data = read_data_kvasir_seg(val_data_path, test_ratio=0.5)
        num_classes = 8
    elif args.dataset == "covid19":
        num_classes = 2
        train_class_data, val_class_data, test_class_data = read_train_test_data_covid19_class()
    else:
        print("dataset error")
        exit(0)

    # ---- 数据变换 ----
    img_size = 224
    data_transform = A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(img_size, img_size),
        A.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ToTensorV2()
    ])
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    print("load model path:",args.model_path)
    # ---- 模型加载 ----
    if "dffv1_base" in args.model_path:
        model = Init_Setting_DFFV1_Base(args.model_path, num_classes=num_classes)
    elif "dffv1_small" in args.model_path:
        model = Init_Setting_DFFV1_Small(args.model_path, num_classes=num_classes)
    elif "dffv1_tiny" in args.model_path:
        model = Init_Setting_DFFV1_Tiny(args.model_path, num_classes=num_classes)
    elif "dffv1_mini" in args.model_path:
        model = Init_Setting_DFFV1_Mini(args.model_path, num_classes=num_classes)
    elif "dffv3_base" in args.model_path:
        model = Init_Setting_DFFV3_Base(args.model_path, num_classes=num_classes)
    elif "hifuse" in args.model_path:
        model = Init_Setting_HiFuse_Small(args.model_path, num_classes=num_classes)
    elif "resnet18" in args.model_path:
        model = Init_Setting_ResNet18(args.model_path, num_classes=num_classes)
    elif "vgg19" in args.model_path:
        model = Init_Setting_VGG19(args.model_path, num_classes=num_classes)
    elif "mixer" in args.model_path:
        model = Init_Setting_MIXER(args.model_path, num_classes=num_classes)
    elif "deit" in args.model_path:
        model = Init_Setting_DEIT(args.model_path, num_classes=num_classes)
    elif "vit_b_32" in args.model_path:
        model = Init_Setting_VIT_32(args.model_path, num_classes=num_classes)
    elif "vit_b_16" in args.model_path:
        model = Init_Setting_VIT_16(args.model_path, num_classes=num_classes)
    elif "swin" in args.model_path:
        model = Init_Setting_SWIN_B(args.model_path, num_classes=num_classes)
    elif "focal" in args.model_path:
        model = Init_Setting_FOCAL_B(args.model_path, num_classes=num_classes)
    elif "convnext" in args.model_path:
        model = Init_Setting_CONVNEXT_B(args.model_path, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type in model_path: {args.model_path}")

    # ---- 分类推理 ----
    print("\n========== Classification Evaluation ==========")
    if args.dataset == "covid19":
        val_class_dataset = COVID19ClassDataset(split="val", transform=data_transform)
    else:
        val_class_dataset = DataSetMutliTaskClassify(data=test_class_data, transform=data_transform)
    val_loader = torch.utils.data.DataLoader(val_class_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=nw,
                                             collate_fn=val_class_dataset.collate_fn)

    pred_classes_list = []
    labels_list = []
    pred_probs_list = []

    with torch.no_grad():
        data_loader = tqdm(val_loader, file=sys.stdout, desc="Classification eval")
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # DFF/HiFuse 模型返回 (seg_output, class_logits)
            if isinstance(outputs, tuple):
                pred_class_logits = outputs[1]
            elif isinstance(outputs, dict):
                pred_class_logits = outputs.get("logit", list(outputs.values())[0])
            else:
                pred_class_logits = outputs

            probs = torch.softmax(pred_class_logits, dim=1)
            pred_classes = torch.max(probs, dim=1)[1]

            pred_classes_list.extend(pred_classes.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            pred_probs_list.extend(probs.cpu().numpy())

    evaluate_metrics(labels_list, pred_classes_list, pred_probs_list, num_classes)
    show_confusion_matrix(labels_list, pred_classes_list, model)

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM" % (total / 1e6))

    # ============ 分割测试 (仅 DFF 模型) ============
    if "dff" in args.model_path.lower():
        print("\n========== Segmentation Evaluation ==========")
        seg_dataset = DataSetMutliTaskSegment(data=test_seg_data, transform=data_transform)
        seg_loader = torch.utils.data.DataLoader(seg_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=nw,
                                                 collate_fn=seg_dataset.collate_fn)

        evaluate_segmentation(model, seg_loader, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--dataset', default="kvasir")
    parser.add_argument('--model_path', type=str,
                        default="/ai/data/DFDF/model_weight/dffv1_small_0-300_kvasir/best_model.pth")
    opt = parser.parse_args()

    main(opt)
