import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, ignore_index=None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C, H, W] or [B, 1, H, W]
            targets: [B, H, W]
        """
        num_classes = logits.shape[1]

        if num_classes == 1:
            probs = torch.sigmoid(logits)
            targets = targets.unsqueeze(1).float()
        else:
            probs = F.softmax(logits, dim=1)
            # 确保targets是long类型用于one_hot编码
            targets_long = targets.long()
            targets = F.one_hot(targets_long, num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets, dims)
        union = torch.sum(probs + targets, dims)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1. - dice.mean()

class DiceCELoss(nn.Module):
    def __init__(self, dice_weight=1.0, ce_weight=1.0):
        super().__init__()
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss()
        self.dw = dice_weight
        self.cw = ce_weight

    def forward(self, logits, targets):
        # 确保targets是Long类型用于CE损失
        targets_for_ce = targets.long()
        return (
            self.dw * self.dice(logits, targets) +
            self.cw * self.ce(logits, targets_for_ce)
        )


class FocalSegLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits, targets,
            reduction='none',
            ignore_index=self.ignore_index
        )
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss.mean()

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        targets = F.one_hot(targets, logits.shape[1]).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        tp = torch.sum(probs * targets, dims)
        fp = torch.sum(probs * (1 - targets), dims)
        fn = torch.sum((1 - probs) * targets, dims)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return 1 - tversky.mean()

def build_seg_loss(cfg):
    name = cfg.LOSS.SEG.NAME

    if name == "Dice":
        return DiceLoss(
            smooth=cfg.LOSS.SEG.SMOOTH
        )
    elif name == "BCE":
        return nn.BCEWithLogitsLoss()

    elif name == "DiceBCE":
        return DiceBCELoss(
            dice_weight=cfg.LOSS.SEG.DICE_WEIGHT,
            bce_weight=cfg.LOSS.SEG.BCE_WEIGHT
        )
    elif name == "Focal":
        return FocalSegLoss(
            gamma=cfg.LOSS.SEG.FOCAL_GAMMA,
            alpha=cfg.LOSS.SEG.FOCAL_ALPHA
        )

    elif name == "Tversky":
        return TverskyLoss(
            alpha=cfg.LOSS.SEG.TVERSKY_ALPHA,
            beta=cfg.LOSS.SEG.TVERSKY_BETA
        )

    else:
        raise ValueError(f"Unknown segmentation loss: {name}")