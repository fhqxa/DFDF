import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.transforms import get_train_cls_transform
from datasets.isic2018 import ISIC2018ClassDataset
from datasets.kvasir import KvasirClassDataset
from utils.metrics import *


def build_test_dataset(dataset, img_size):
    if dataset == "isic2018":
        return ISIC2018ClassDataset(
            split="val",
            transform=get_train_cls_transform(img_size)
        )
    elif dataset == "kvasir":
        return KvasirClassDataset(
            split="val",
            transform=get_train_cls_transform(img_size)
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


@torch.no_grad()
def test(
        model,
        dataset,
        model_path=None,
        batch_size=64,
        img_size=224,
        device="cuda:0"
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # --- load weights ---


    if model_path is not None:
        # 如果是因为DataParallel导致的键名前缀问题
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['net']

        # 如果键名有"module."前缀，需要移除
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # 移除"module."前缀
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=False)  # 使用strict=False
    model.to(device)
    model.eval()

    # --- dataset & loader ---
    test_ds = build_test_dataset(dataset, img_size)
    print(f"[Test] dataset={dataset}", len(test_ds))
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    task_type = model.task_type

    # 存储所有预测和真实标签
    all_preds = []
    all_labels = []
    all_masks = []
    all_targets = []

    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images, train_task="joint")

        if task_type == "seg":
            # 存储分割预测和目标
            all_masks.append(outputs["mask"].cpu())
            all_targets.append(labels.cpu())
        else:
            # 存储分类预测和标签
            all_preds.append(outputs["logit"].argmax(dim=1).cpu())
            all_labels.append(labels.cpu())

    # 拼接所有批次的结果
    if task_type == "seg":
        all_masks = torch.cat(all_masks, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # 计算分割指标
        metrics = compute_segmentation_metrics(all_masks, all_targets)
        print_segmentation_results(metrics, dataset)
        return metrics["dice"]

    else:
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # 计算分类指标
        metrics = compute_classification_metrics(all_preds.numpy(), all_labels.numpy())
        print_classification_results(metrics, dataset)

    return metrics['acc']


if __name__ == "__main__":
    import argparse
    from model.dff import DFF_S  # 假设这是你的模型类

    parser = argparse.ArgumentParser(description="Test model on specified dataset")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model checkpoint")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["isic2018", "kvasir"],
                        help="Dataset to test on")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for testing")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Input image size")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run testing on")

    args = parser.parse_args()

    # 假设模型初始化需要num_classes参数
    # 这里可能需要根据具体模型调整
    model = DFF_S(num_classes=7)  # 根据数据集调整类别数

    # 执行测试
    result = test(
        model=model,
        dataset=args.dataset,
        model_path=args.model_path,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=args.device
    )

    print(f"Test completed. Result: {result}")
