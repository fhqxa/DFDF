import os
import glob
import random
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


class KvasirConfig:
    root = "/ai/data/Kvasir"

    class_root = "kvasir-dataset"
    seg_root = "kvasir-seg"

class KvasirClassDataset(Dataset):
    def __init__(self, split="train", transform=None, seed=52):
        self.transform = transform
        random.seed(seed)

        root = os.path.join(KvasirConfig.root, KvasirConfig.class_root)
        classes = sorted(os.listdir(root))

        self.samples = []

        for idx, cls in enumerate(classes):
            imgs = sorted(glob.glob(os.path.join(root, cls, "*")))
            random.shuffle(imgs)

            mid = len(imgs) // 2
            imgs = imgs[:mid] if split == "train" else imgs[mid:]

            for img in imgs:
                self.samples.append((img, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        image = sitk.GetArrayFromImage(sitk.ReadImage(path))

        # --- 关键：保证 HWC ---
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))

        image = image.astype(np.float32)


class KvasirSegDataset(Dataset):
    def __init__(self, split="train", transform=None, test_ratio=0.1, seed=42):
        self.transform = transform
        random.seed(seed)

        mask_dir = os.path.join(
            KvasirConfig.root, KvasirConfig.seg_root, "masks"
        )
        masks = glob.glob(os.path.join(mask_dir, "*.jpg"))
        random.shuffle(masks)

        split_idx = int(len(masks) * test_ratio)
        masks = masks[split_idx:] if split == "train" else masks[:split_idx]

        self.samples = [
            (m.replace("masks", "images"), m) for m in masks
        ]

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
        mask = (mask > 0).astype(np.float32)  # H x W

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image = out["image"]
            mask = out["mask"]  # 已经是 [1, H, W]

        return image, mask

