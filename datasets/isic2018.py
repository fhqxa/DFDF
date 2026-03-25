import os
import glob
import pandas as pd
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


class ISIC2018Config:
    root = "/home/zy/lwc/ISIC2018"

    cls_csv = {
        "train": "ISIC2018_Task3_Training_GroundTruth.csv",
        "val": "ISIC2018_Task3_Test_GroundTruth.csv",
    }

    cls_img_dir = {
        "train": "ISIC2018_Task3_Training_Input",
        "val": "ISIC2018_Task3_Test_Input",
    }

    seg_img_dir = {
        "train": "ISIC2018_Task1-2_Training_Input",
        "val": "ISIC2018_Task1-2_Test_Input",
    }

    seg_mask_dir = {
        "train": "ISIC2018_Task1_Training_GroundTruth",
        "val": "ISIC2018_Task1_Test_GroundTruth",
    }

    classes = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


class ISIC2018ClassDataset(Dataset):
    # def __init__(self, split="train", transform=None):
    #     self.transform = transform
    #     cfg = ISIC2018Config
    #
    #     df = pd.read_csv(os.path.join(cfg.root, cfg.cls_csv[split]))
    #     df = df.sample(frac=1, random_state=42)
    #
    #     self.samples = []
    #     for _, row in df.iterrows():
    #         label = row[cfg.classes].values.argmax()
    #         img_path = os.path.join(
    #             cfg.root, cfg.cls_img_dir[split], f"{row['image']}.jpg"
    #         )
    #         self.samples.append((img_path, label))
    #
    #     # 记录类别数量信息
    #     self.num_classes = len(cfg.classes)
    #     self.labels = [sample[1] for sample in self.samples]
    def __init__(self, split="train", transform=None):
        self.transform = transform
        cfg = ISIC2018Config

        df = pd.read_csv(os.path.join(cfg.root, cfg.cls_csv['train']))
        df = df.sample(frac=1, random_state=42)

        self.samples = []
        for _, row in df.iterrows():
            label = row[cfg.classes].values.argmax()
            img_path = os.path.join(
                cfg.root, cfg.cls_img_dir['train'], f"{row['image']}.jpg"
            )
            self.samples.append((img_path, label))

        if split == 'train':
            self.samples = self.samples[:int(len(self.samples)*0.7)]
        if split == 'val':
            self.samples = self.samples[int(len(self.samples)*0.7):]

        # 记录类别数量信息
        self.num_classes = len(cfg.classes)
        self.labels = [sample[1] for sample in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # --- 关键修改 1：确保 HWC + float32 ---
        image = sitk.GetArrayFromImage(sitk.ReadImage(path))
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        image = image.astype(np.float32)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, label

    def get_cls_num_list(self):
        """获取每个类别的样本数量列表"""
        cls_num_list = [0] * self.num_classes
        for label in self.labels:
            cls_num_list[label] += 1
        return cls_num_list


class ISIC2018SegDataset(Dataset):
    def __init__(self, split="train", transform=None):
        self.transform = transform
        cfg = ISIC2018Config

        mask_dir = os.path.join(cfg.root, cfg.seg_mask_dir[split])
        self.samples = []

        for m in glob.glob(os.path.join(mask_dir, "*.png")):
            img = (
                m.replace(cfg.seg_mask_dir[split], cfg.seg_img_dir[split])
                 .replace("_segmentation", "")
                 .replace(".png", ".jpg")
            )
            self.samples.append((img, m))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p, mask_p = self.samples[idx]

        # --- image ---
        image = sitk.GetArrayFromImage(sitk.ReadImage(img_p))
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        image = image.astype(np.float32)

        # --- mask ---
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_p))
        mask = (mask > 0).astype(np.float32)   # H x W

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image = out["image"]
            mask = out["mask"]    # 已是 [1, H, W]

        return image, mask

