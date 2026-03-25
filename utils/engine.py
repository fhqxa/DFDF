# utils/engine.py
import torch
from tqdm import tqdm
from utils.metrics import binary_iou, binary_dice
import math


def get_train_weights(cfg, epoch, strategy):
    """
    根据策略返回分类和分割任务的权重

    Args:
        cfg: 配置对象
        epoch: 当前训练轮数
        strategy: 权重策略 (如 "consist-1.0", "cos", "warmup", "linear" 等)

    Returns:
        list: [cls_weight, seg_weight]
    """
    if strategy.startswith("consist-"):
        # 固定权重策略
        cls_weight = float(strategy.split("-")[1])
        seg_weight = 1.0 - cls_weight
    elif strategy == "cos":
        # 余弦衰减策略
        cls_weight = 0.5 * (1 + math.cos(math.pi * epoch / cfg.TRAIN.EPOCHS))
        seg_weight = 1.0 - cls_weight
    elif strategy == "warmup":
        # 预热策略
        if epoch < cfg.TRAIN.EPOCHS // 2:
            cls_weight = epoch / (cfg.TRAIN.EPOCHS // 2)
        else:
            cls_weight = 1.0
        seg_weight = 1.0 - cls_weight
    elif strategy == "linear":
        # 线性变化策略
        cls_weight = epoch / cfg.TRAIN.EPOCHS
        seg_weight = 1.0 - cls_weight
    else:
        # 默认策略
        cls_weight = 0.5
        seg_weight = 0.5

    return [cls_weight, seg_weight]


def train_one_epoch(
        model,
        data_loader,
        optimizer,
        scheduler,
        device,
        epoch,
        task_type,
        scaler,
        cls_loss,
        seg_loss,
        logger,  # 添加 logger 参数
        cfg
):
    assert task_type in ["seg", "cls"]
    if "V" not in cfg.MODEL.NAME:
        model.freeze_for_task(task_type)
    model.train()
    total_loss = 0.0
    total_metric = 0.0
    total_samples = 0

    data_loader = tqdm(data_loader, desc=f"[Train][{task_type}] Epoch {epoch}")

    for images, targets in data_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        bs = images.size(0)
        total_samples += bs

        optimizer.zero_grad(set_to_none=True)
        cls_weight, seg_weight = get_train_weights(cfg, epoch, cfg.TRAIN.STRATEGY)

        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            pred = model(images, task_type)
            if task_type == "seg":
                loss = seg_loss(pred['seg_mask'], targets)
                loss = loss * seg_weight
                metric = binary_iou(pred['seg_mask'], targets)
            else:
                loss = cls_loss(pred['class_logit'], targets)
                loss = loss * cls_weight
                metric = (pred['class_logit'].argmax(1) == targets).float().mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * bs
        total_metric += metric.item() * bs

        data_loader.set_postfix(
            loss=total_loss / total_samples,
            metric=total_metric / total_samples,
            lr=optimizer.param_groups[0]["lr"],
            cls_weight=cls_weight,
            seg_weight=seg_weight
        )

    # 计算平均损失和指标
    avg_loss = total_loss / total_samples
    avg_metric = total_metric / total_samples

    # 写入 TensorBoard 日志
    if task_type == "seg":
        logger._writer.add_scalar(f'Train/{task_type}_loss', avg_loss, epoch)
        logger._writer.add_scalar(f'Train/{task_type}_metric', avg_metric, epoch)
    else:
        logger._writer.add_scalar(f'Train/{task_type}_loss', avg_loss, epoch)
        logger._writer.add_scalar(f'Train/{task_type}_accuracy', avg_metric, epoch)


def train_joint_one_epoch(
        model,
        data_loader,
        optimizer,
        scheduler,
        device,
        epoch,
        scaler,
        cls_loss,
        seg_loss,
        logger,  # 添加 logger 参数
        cfg
):
    if "V" not in cfg.MODEL.NAME:
        model.freeze_for_task("joint")
        task_type = model.task_type
    else:
        task_type = "cls"
    model.train()
    total_loss = 0.0
    total_metric = 0.0
    total_samples = 0

    data_loader = tqdm(data_loader, desc=f"[Train Joint][{task_type}] Epoch {epoch}")

    for images, targets in data_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        bs = images.size(0)
        total_samples += bs

        optimizer.zero_grad(set_to_none=True)
        cls_weight, seg_weight = get_train_weights(cfg, epoch, cfg.TRAIN.STRATEGY)

        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            pred = model(images, "joint")

            if task_type == "seg":
                loss = seg_loss(pred['mask'], targets)
                loss = loss * seg_weight
                metric = binary_iou(pred['mask'], targets)
            else:
                loss = cls_loss(pred['logit'], targets)
                loss = loss * cls_weight
                metric = (pred['logit'].argmax(1) == targets).float().mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * bs
        total_metric += metric.item() * bs

        data_loader.set_postfix(
            loss=total_loss / total_samples,
            metric=total_metric / total_samples,
            lr=optimizer.param_groups[0]["lr"]
        )

    # 计算平均损失和指标
    avg_loss = total_loss / total_samples
    avg_metric = total_metric / total_samples

    # 写入 TensorBoard 日志
    if task_type == "seg":
        logger._writer.add_scalar(f'Train Joint/{task_type}_loss', avg_loss, epoch)
        logger._writer.add_scalar(f'Train Joint/{task_type}_metric', avg_metric, epoch)
    else:
        logger._writer.add_scalar(f'Train Joint/{task_type}_loss', avg_loss, epoch)
        logger._writer.add_scalar(f'Train Joint/{task_type}_accuracy', avg_metric, epoch)
