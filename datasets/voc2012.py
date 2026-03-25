import glob
import os
import random


def read_data_voc2012_class(root: str):
    assert os.path.exists(root), f"dataset root: {root} does not exist."

    # 获取20种标签的名称并按字符排序
    label_files = sorted([f for f in os.listdir(os.path.join(root, "ImageSets", "Main")) if "_trainval.txt" in f])
    labels = [lf.split("_")[0] for lf in label_files]
    print(labels)
    train_set = []
    test_set = []

    # 提前读取所有标签文件内容
    label_contents = {}
    for label_file in label_files:
        file_path = os.path.join(root, "ImageSets/Main", label_file)
        with open(file_path, 'r') as lf:
            label_contents[label_file] = {line.strip().split()[0]: line.strip().split()[1] for line in lf if
                                          len(line.strip().split()) == 2}

    # 读取train.txt和val.txt文件
    for file_name, dataset in [("train.txt", train_set), ("val.txt", test_set)]:
        file_path = os.path.join(root, "ImageSets/Main", file_name)
        assert os.path.exists(file_path), f"{file_name} does not exist."

        with open(file_path, 'r') as f:
            image_ids = [line.strip() for line in f.readlines()]

        for image_id in image_ids:
            image_path = os.path.join(root, "JPEGImages", f"{image_id}.jpg")
            one_hot_label = [0] * len(labels)  # 初始化为20个元素的全零one-hot标签

            # 更新one-hot标签
            for idx, label_file in enumerate(label_files):
                if image_id in label_contents[label_file] and label_contents[label_file][image_id] == "1":
                    one_hot_label[idx] = 1

            dataset.append({"image": image_path, "label": one_hot_label})

    print(f"{len(train_set)} images for training dataset.")
    print(f"{len(test_set)} images for testing dataset.")

    return train_set, test_set

def read_data_voc2012_seg(root: str, input_folder: str = "images", mask_folder: str = "masks", test_ratio: float = 0.1):
    assert os.path.exists(root), f"dataset root: {root} does not exist."

    train_set = []
    test_set = []

    # 读取train.txt和val.txt文件
    for file_name, dataset in [("train.txt", train_set), ("val.txt", test_set)]:
        file_path = os.path.join(root, "ImageSets/Segmentation", file_name)

        with open(file_path, 'r') as f:
            image_ids = [line.strip() for line in f.readlines()]

        for image_id in image_ids:
            image_path = os.path.join(root, "JPEGImages", f"{image_id}.jpg")
            mask_path = os.path.join(root, "SegmentationClass", f"{image_id}.png")
            dataset.append({"image": image_path, "mask": mask_path})

    print(f"{len(train_set)} images for training dataset.")
    print(f"{len(test_set)} images for testing dataset.")

    return train_set, test_set

