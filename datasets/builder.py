from datasets.isic2018 import (
    ISIC2018ClassDataset,
    ISIC2018SegDataset,
)
from datasets.kvasir import (
    KvasirClassDataset,
    KvasirSegDataset,
)

from datasets.covid19 import (
    COVID19ClassDataset,
    COVID19SegDataset,
)


def build_dataset(
    dataset_name: str,
    task: str,
    split: str,
    transform=None,
):
    """
    dataset_name: 'isic2018' | 'kvasir'
    task: 'class' | 'seg'
    split: 'train' | 'val'
    """

    if dataset_name == "isic2018":
        if task == "class":
            return ISIC2018ClassDataset(split=split, transform=transform)
        elif task == "seg":
            return ISIC2018SegDataset(split=split, transform=transform)

    if dataset_name == "kvasir":
        if task == "class":
            return KvasirClassDataset(split=split, transform=transform)
        elif task == "seg":
            return KvasirSegDataset(split=split, transform=transform)

    if dataset_name == "covid19":
        if task == "class":
            return COVID19ClassDataset(split=split, transform=transform)
        elif task == "seg":
            return COVID19SegDataset(split=split, transform=transform)

    raise ValueError(f"Unknown dataset/task: {dataset_name}, {task}")
