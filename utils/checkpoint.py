# utils/checkpoint.py
import os
import torch


def save_checkpoint(
    path,
    model,
    epoch,
    optimizer=None,
    scheduler_seg=None,
    scheduler_cls=None
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "net": model.state_dict(),
            "epoch": epoch,
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler_seg": scheduler_seg.state_dict() if scheduler_seg else None,
            "scheduler_cls": scheduler_cls.state_dict() if scheduler_cls else None,
        },
        path
    )
