import argparse
import os
import sys

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset.dataset import DataSetMask
from dataset.ISIC2018 import DataSetMutliTaskSegment, DataSetMutliTaskClassify, read_train_data_isic2018_class, \
    read_train_data_isic2018_seg, read_test_data_isic2018_class, read_test_data_isic2018_seg
from dataset.Kvasir import read_train_test_data_kvasir_class,read_data_kvasir_seg

from utils import show_confusion_matrix
from sklearn.metrics import confusion_matrix
from analysis.GradCAM import (Init_Setting_DFFV1_Small, Init_Setting_DFFV1_Tiny, Init_Setting_HiFuse_Small,Init_Setting_DFFV1_Base, \
                              Init_Setting_DFFV3_Base,Init_Setting_ResNet18, Init_Setting_MIXER, \
                              Init_Setting_DEIT, Init_Setting_VIT_32, Init_Setting_VGG19, Init_Setting_VIT_16, \
                              Init_Setting_SWIN_B, Init_Setting_FOCAL_B, Init_Setting_CONVNEXT_B, Init_Setting_DFFV1_Mini)


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
    """
    labels_list: 真实标签列表
    pred_classes_list: 预测类别列表
    pred_probs_list: 预测概率列表 (用于计算 AUC)，若为 None 则不计算 AUC
    num_classes: 类别数量 (用于计算多分类 AUC)
    """
    acc = accuracy_score(labels_list, pred_classes_list)
    prec = precision_score(labels_list, pred_classes_list, average='macro', zero_division=0)
    rec = recall_score(labels_list, pred_classes_list, average='macro', zero_division=0)
    f1 = f1_score(labels_list, pred_classes_list, average='macro')

    # 计算 MCC
    mcc = matthews_corrcoef(labels_list, pred_classes_list)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"MCC: {mcc:.4f}")

    auc = None
    if pred_probs_list is not None and num_classes is not None:
        try:
            # 多分类 AUC 通常使用 One-vs-Rest (OvR) 策略并取 macro average
            # 需要将 labels 转换为 one-hot 或者直接使用 multiclass 参数 (sklearn >= 0.22)
            auc = roc_auc_score(labels_list, pred_probs_list, average='macro', multi_class='ovr')
            print(f"AUC (Macro): {auc:.4f}")
        except ValueError as e:
            print(f"Could not calculate AUC: {e}")

    return acc, prec, rec, f1, mcc, auc


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    # 加载验证数据
    if args.dataset == "isic2018":
        val_data_path = "/ai/data/ISIC2018"
        test_class_data = read_test_data_isic2018_class(val_data_path)
    elif args.dataset == "kvasir":
        val_data_path = "/ai/data/Kvasir"
        train_class_data, test_class_data = read_train_test_data_kvasir_class(val_data_path)
    elif args.dataset == "covid19":
        num_classes = 2
        train_class_data, val_class_data, test_class_data = read_train_test_data_covid19_class()
    else:
        print("dataset error")
        exit(0)

    # 数据变换
    img_size = 224
    data_transform = A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(img_size, img_size),
        A.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ToTensorV2()
    ])

    val_class_dataset = DataSetMutliTaskClassify(data=test_class_data, transform=data_transform)

    # 数据加载器
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    val_loader = torch.utils.data.DataLoader(val_class_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=nw,
                                             collate_fn=val_class_dataset.collate_fn)

    # 模型加载
    if args.dataset == "isic2018":
        num_classes = 7
    elif args.dataset == "kvasir":
        num_classes = 8


    model = Init_Setting_HiFuse_Small(args.model_path, num_classes=num_classes)
    # model = Init_Setting_ResNet18(args.model_path,num_classes=num_classes)
    # model = Init_Setting_VGG19(args.model_path,num_classes=num_classes)
    # model = Init_Setting_MIXER(args.model_path,num_classes=num_classes)
    # model = Init_Setting_DEIT(args.model_path, num_classes=num_classes)
    # model = Init_Setting_VIT_32(args.model_path, num_classes=num_classes)
    # model = Init_Setting_VIT_16(args.model_path, num_classes=num_classes)
    # model = Init_Setting_CONVNEXT_B(args.model_path, num_classes=num_classes)
    # model = Init_Setting_FOCAL_B(args.model_path, num_classes=num_classes)
    # model = Init_Setting_SWIN_B(args.model_path, num_classes=num_classes)



    # 推理与评估
    pred_classes_list = []
    labels_list = []
    pred_probs_list = []  # 新增：用于存储概率

    with torch.no_grad():
        data_loader = tqdm(val_loader, file=sys.stdout)
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            if isinstance(outputs, tuple):
                pred_class_logits = outputs[1]  # 取出第二个输出 (logits)
            else:
                pred_class_logits = outputs

            probs = torch.softmax(pred_class_logits, dim=1)
            pred_classes = torch.max(probs, dim=1)[1]

            pred_classes_list.extend(pred_classes.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            pred_probs_list.extend(probs.cpu().numpy())

    # 计算评价指标
    evaluate_metrics(labels_list, pred_classes_list, pred_probs_list, num_classes)

    # 展示混淆矩阵
    show_confusion_matrix(labels_list, pred_classes_list, model)

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM" % (total / 1e6))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=24)
    # parser.add_argument('--dataset', default="isic2018")
    parser.add_argument('--dataset', default="kvasir")
    parser.add_argument('--model_path', type=str, default="/ai/data/DFDF/model_weight/dffv1_small_0-300_kvasir/best_model.pth")
    opt = parser.parse_args()

    main(opt)


# python analysis.py --dataset kvasir --model_path
