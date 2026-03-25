import os
import argparse
import random
import numpy as np
import torch

from datasets.builder import build_dataset
from datasets.transforms import (
    get_train_seg_transform,
    get_train_cls_transform,
)

from model.dff import DFF_S, DFF_T, DFF_B
from model.dffv1 import DFFV1_S, DFFV1_T, DFFV1_B
from utils.optim import create_lr_scheduler, get_params_groups
from utils.engine import train_one_epoch, train_joint_one_epoch
from utils.checkpoint import save_checkpoint
from utils.cls_losses import build_cls_loss
from utils.seg_losses import build_seg_loss
from utils.config import get_cfg
from test import test
# 在 main 函数中初始化 logger
from utils.logger import setup_logger

cfg = get_cfg()

def main(args):


    # -------- merge yaml --------
    if args.data:
        cfg.merge_from_file(os.path.join("configs/data", args.data + ".yaml"))
    if args.model:
        cfg.merge_from_file(os.path.join("configs/model", args.model + ".yaml"))

    # -------- CLI override --------
    args.opts = [item.strip() for item in args.opts]
    cfg.merge_from_list(args.opts)
    # -------- output dir --------
    if cfg.OUTPUT_DIR is None:
        cfg_name = f"{cfg.DATA.NAME}_{cfg.MODEL.NAME}"
        opts_name = "".join(["_" + item.replace(".", "-") for item in args.opts])
        cfg.OUTPUT_DIR = os.path.join("./output", cfg_name + opts_name)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    logger = setup_logger(cfg.OUTPUT_DIR)

    print("************ CONFIG ************")
    print(cfg)
    print("********************************")

    # -------- seed --------
    if cfg.TRAIN.SEED is not None:
        seed = cfg.TRAIN.SEED
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    device = torch.device(cfg.TRAIN.DEVICE if torch.cuda.is_available() else "cpu")
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    # -------- dataset --------
    train_seg_ds = build_dataset(
        cfg.DATA.NAME, "seg", "train",
        get_train_seg_transform(cfg.DATA.IMG_SIZE)
    )
    train_cls_ds = build_dataset(
        cfg.DATA.NAME, "class", "train",
        get_train_cls_transform(cfg.DATA.IMG_SIZE)
    )
    cls_num_list = train_cls_ds.get_cls_num_list()

    train_seg_loader = torch.utils.data.DataLoader(
        train_seg_ds, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True
    )
    train_cls_loader = torch.utils.data.DataLoader(
        train_cls_ds, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True
    )

    # -------- model --------
    if cfg.MODEL.NAME == "DFF_T":
        model = DFF_T(num_classes=cfg.MODEL.NUM_CLASSES).to(device)
    elif cfg.MODEL.NAME == "DFF_B":
        model = DFF_B(num_classes=cfg.MODEL.NUM_CLASSES).to(device)
    elif cfg.MODEL.NAME == "DFF_S":
        model = DFF_S(num_classes=cfg.MODEL.NUM_CLASSES).to(device)
    elif cfg.MODEL.NAME == "DFFV1_T":
        model = DFFV1_T(num_classes=cfg.MODEL.NUM_CLASSES).to(device)
    elif cfg.MODEL.NAME == "DFFV1_B":
        model = DFFV1_B(num_classes=cfg.MODEL.NUM_CLASSES).to(device)
    elif cfg.MODEL.NAME == "DFFV1_S":
        model = DFFV1_S(num_classes=cfg.MODEL.NUM_CLASSES).to(device)
    else:
        raise ValueError(f"Unknown model: {cfg.MODEL.NAME}")

    optimizer = torch.optim.AdamW(
        get_params_groups(model, cfg.OPTIMIZER.WEIGHT_DECAY),
        lr=cfg.OPTIMIZER.LR
    )

    scheduler_seg = create_lr_scheduler(
        optimizer, len(train_seg_loader), cfg.TRAIN.EPOCHS
    )
    scheduler_cls = create_lr_scheduler(
        optimizer, len(train_cls_loader), cfg.TRAIN.EPOCHS
    )

    cls_loss = build_cls_loss(cfg, cls_num_list, device)
    seg_loss = build_seg_loss(cfg)

    best_acc = 0.0

    # -------- train loop --------
    for epoch in range(1, cfg.TRAIN.EPOCHS + 1):
        train_one_epoch(
            model, train_seg_loader, optimizer, scheduler_seg,
            device, epoch, "seg", scaler,
            cls_loss, seg_loss, logger, cfg
        )

        # train_one_epoch(
        #     model, train_cls_loader, optimizer, scheduler_cls,
        #     device, epoch, "cls", scaler,
        #     cls_loss, seg_loss, logger, cfg
        # )

        train_joint_one_epoch(
            model, train_cls_loader, optimizer, scheduler_cls,
            device, epoch, scaler,
            cls_loss, seg_loss, logger, cfg
        )

        if cfg.TEST.DO_TEST and epoch % cfg.TEST.INTERVAL == 0:
            acc = test(model, cfg.DATA.NAME)
            if acc > best_acc:
                best_acc = acc
                save_checkpoint(
                    os.path.join(cfg.OUTPUT_DIR, "best.pth"),
                    model, epoch
                )

            print(f"[Epoch {epoch}] Acc: {acc:.4f}")

    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, default="")
    parser.add_argument("--model", "-m", type=str, default="")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line"
    )
    args = parser.parse_args()
    main(args)
