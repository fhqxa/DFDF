import os
import glob
import random
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


class COVID19Config:
    # 请根据实际情况修改根目录路径
    root = "/path/to/your/covid19_dataset"

    # 分类任务目录结构
    cls_root = "COVID-19 CT"
    cls_subfolders = {
        "negative": "CT_NonCOVID",  # 标签 0
        "positive": "CT_COVID"  # 标签 1
    }

    # 分割任务目录结构
    seg_root = "COVID-19 CT segmentation"
    seg_img_dir = "frames"
    seg_mask_dir = "masks"

    # 数据集划分比例
    split_ratios = {
        "train": 0.60,
        "val": 0.15,
        "test": 0.25
    }


class COVID19ClassDataset(Dataset):
    def __init__(self, split="train", transform=None, seed=42):
        """
        Args:
            split (str): 'train', 'val', or 'test'
            transform: Albumentations transform
            seed (int): Random seed for reproducible splitting
        """
        self.transform = transform
        self.split = split
        cfg = COVID19Config

        random.seed(seed)

        self.samples = []
        self.labels = []
        self.num_classes = 2  # NonCOVID (0), COVID (1)

        # 1. 收集所有数据
        all_data = []

        # 处理 Negative 类 (标签 0)
        neg_dir = os.path.join(cfg.root, cfg.cls_root, cfg.cls_subfolders["negative"])
        if os.path.exists(neg_dir):
            neg_files = sorted([os.path.join(neg_dir, f) for f in os.listdir(neg_dir)
                                if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))])
            all_data.extend([(f, 0) for f in neg_files])

        # 处理 Positive 类 (标签 1)
        pos_dir = os.path.join(cfg.root, cfg.cls_root, cfg.cls_subfolders["positive"])
        if os.path.exists(pos_dir):
            pos_files = sorted([os.path.join(pos_dir, f) for f in os.listdir(pos_dir)
                                if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))])
            all_data.extend([(f, 1) for f in pos_files])

        # 2. 打乱并划分数据集
        random.shuffle(all_data)
        total_len = len(all_data)

        train_end = int(total_len * cfg.split_ratios["train"])
        val_end = train_end + int(total_len * cfg.split_ratios["val"])

        if split == "train":
            self.samples = all_data[:train_end]
        elif split == "val":
            self.samples = all_data[train_end:val_end]
        elif split == "test":
            self.samples = all_data[val_end:]
        else:
            raise ValueError(f"Invalid split: {split}")

        # 提取标签列表用于统计
        self.labels = [item[1] for item in self.samples]

        print(f"COVID19 Class Dataset [{split}]: {len(self.samples)} samples loaded.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # 读取图像
        image = sitk.GetArrayFromImage(sitk.ReadImage(path))

        # 确保格式为 HWC (Height, Width, Channel)
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        elif image.ndim == 2:
            # 如果是灰度图，扩展为 3 通道以适配常见模型输入
            image = np.stack([image] * 3, axis=-1)

        image = image.astype(np.float32)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, label

    def get_cls_num_list(self):
        """获取每个类别的样本数量列表 [NonCOVID_count, COVID_count]"""
        cls_num_list = [0] * self.num_classes
        for label in self.labels:
            cls_num_list[label] += 1
        return cls_num_list


class COVID19SegDataset(Dataset):
    def __init__(self, split="train", transform=None, seed=42):
        """
        Args:
            split (str): 'train', 'val', or 'test'
            transform: Albumentations transform (expects image and mask)
            seed (int): Random seed
        """
        self.transform = transform
        self.split = split
        cfg = COVID19Config

        random.seed(seed)

        # 1. 获取所有匹配的图像和掩码对
        mask_dir = os.path.join(cfg.root, cfg.seg_root, cfg.seg_mask_dir)
        img_dir = os.path.join(cfg.root, cfg.seg_root, cfg.seg_img_dir)

        mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")) +
                            glob.glob(os.path.join(mask_dir, "*.jpg")))

        self.all_samples = []
        for mask_path in mask_files:
            filename = os.path.basename(mask_path)
            # 假设 frames 和 masks 文件名完全一致，如果后缀不同需在此调整
            img_path = os.path.join(img_dir, filename)

            if os.path.exists(img_path):
                self.all_samples.append((img_path, mask_path))
            else:
                print(f"Warning: Image not found for mask {filename}, skipping.")

        # 2. 打乱并划分
        random.shuffle(self.all_samples)
        total_len = len(self.all_samples)

        train_end = int(total_len * cfg.split_ratios["train"])
        val_end = train_end + int(total_len * cfg.split_ratios["val"])

        if split == "train":
            self.samples = self.all_samples[:train_end]
        elif split == "val":
            self.samples = self.all_samples[train_end:val_end]
        elif split == "test":
            self.samples = self.all_samples[val_end:]
        else:
            raise ValueError(f"Invalid split: {split}")

        print(f"COVID19 Seg Dataset [{split}]: {len(self.samples)} samples loaded.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p, mask_p = self.samples[idx]

        # --- 读取图像 ---
        image = sitk.GetArrayFromImage(sitk.ReadImage(img_p))
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        elif image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        image = image.astype(np.float32)

        # --- 读取掩码 ---
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_p))

        # 处理可能的多通道掩码 (如 RGB)，转为单通道灰度
        if mask.ndim == 3 and mask.shape[-1] == 3:
            mask = np.mean(mask, axis=-1)

        # 二值化：非零区域视为病灶 (1), 否则为背景 (0)
        mask = (mask > 0).astype(np.float32)

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image = out["image"]
            mask = out["mask"]
            # 确保 mask 有通道维度 [1, H, W]，如果 transform 没加的话
            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=0)

        return image, mask
