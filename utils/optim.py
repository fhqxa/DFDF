# utils/optim.py
import math
import torch


def create_lr_scheduler(
    optimizer,
    num_step,
    epochs,
    warmup=True,
    warmup_epochs=1,
    warmup_factor=1e-3,
    end_factor=1e-2
):
    def f(x):
        if warmup and x < warmup_epochs * num_step:
            alpha = x / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            current = x - warmup_epochs * num_step
            total = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(math.pi * current / total)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model, weight_decay=1e-5):
    decay, no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
