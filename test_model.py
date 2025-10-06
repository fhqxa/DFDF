import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torchvision.models import resnet50
import torchvision.models as models
from torch.nn.functional import interpolate
from sklearn.metrics import accuracy_score
import numpy as np
from model.ddfv2 import DFFV2_Small as DFF


def calculate_pixel_accuracy(preds, targets):
    """Calculate pixel accuracy for segmentation task."""
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0

def calculate_iou(preds, targets, num_classes):
    """Calculate IoU for segmentation task."""
    ious = []
    preds = preds.view(-1)
    targets = targets.view(-1)

    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = targets == cls
        intersection = (pred_inds & target_inds).sum().float().item()
        union = (pred_inds | target_inds).sum().float().item()

        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(intersection / union)

    return np.nanmean(ious)  # Ignore NaN values when calculating mean


# 模拟训练数据
def generate_dummy_data(batch_size, num_classes, img_size):
    X = torch.rand(batch_size, 3, img_size, img_size)  # 随机生成图像
    y_class = torch.randint(0, num_classes, (batch_size,))  # 随机生成分类标签
    y_seg = torch.randint(0, 2, (batch_size, 1, img_size, img_size))  # 随机生成分割mask
    return X, y_class, y_seg


if __name__ == '__main__':
    # 模拟数据集
    img_size = 224
    num_classes = 1
    batch_size = 8
    data_size = 100

    X, y_class, y_seg = generate_dummy_data(data_size, num_classes, img_size)
    dataset = TensorDataset(X, y_class, y_seg)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    model = DFF(num_classes,1).cuda()

    # 定义损失函数
    criterion_class = nn.CrossEntropyLoss()  # 分类损失
    criterion_seg = nn.BCELoss()  # 分割损失

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 模拟训练
    num_epochs = 1
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        all_pixel_accuracies = []
        all_ious = []
        for batch_X, batch_y_class, batch_y_seg in data_loader:
            batch_X = batch_X.cuda()
            batch_y_class = batch_y_class.cuda()
            batch_y_seg = batch_y_seg.cuda()

            # 前向传播
            tumor_output, seg_output = model(batch_X)
            break

            # 计算损失
            loss_class = criterion_class(tumor_output, batch_y_class)
            loss_seg = criterion_seg(seg_output, batch_y_seg.float())
            loss = loss_class + loss_seg

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            seg_preds = (seg_output > 0.5).float()  # Assuming binary segmentation, threshold at 0.5
            seg_targets = batch_y_seg.float()

            pixel_accuracy = calculate_pixel_accuracy(seg_preds, seg_targets)
            iou = calculate_iou(seg_preds, seg_targets, num_classes=2)  # Binary segmentation has 2 classes: 0 and 1

            all_pixel_accuracies.append(pixel_accuracy)
            all_ious.append(iou)

        avg_loss = total_loss / 800
        avg_pixel_accuracy = np.mean(all_pixel_accuracies)
        avg_iou = np.nanmean(all_ious)  # Handle NaN values from empty classes
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Pixel Accuracy: {avg_pixel_accuracy:.4f}, IoU: {avg_iou:.4f}")


    print("训练完成！")
