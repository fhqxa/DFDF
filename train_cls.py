import os
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset.ISIC2018 import DataSetMutliTaskSegment, DataSetMutliTaskClassify, read_train_data_isic2018_class, \
    read_train_data_isic2018_seg, read_test_data_isic2018_class, read_test_data_isic2018_seg
from dataset.Glas import read_train_data_glas_class,read_train_data_glas_seg, read_test_data_glas_class, read_test_data_glas_seg
from dataset.Kvasir import read_train_test_data_kvasir_class,read_data_kvasir_seg
from dataset.voc2012 import read_data_voc2012_class,read_data_voc2012_seg
from model.hifuse import HiFuse_Small
from torchvision.models import resnet18 as resnet18_torch
from torchvision.models import vgg19, vit_b_16, vit_b_32
from utils import create_lr_scheduler, get_params_groups, train_only_class, evaluate_only_class, save_checkpoint


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    task_name = f"{args.model}_{args.dataset}"
    torch.cuda.empty_cache()
    print(f"using {device} device.")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')

    tb_writer = SummaryWriter(f"runs/{task_name }")

    if args.dataset == "isic2018":
        data_path = "/home/dj212/lwc/data/ISIC2018"
        # train_seg_data = read_train_data_isic2018_seg(args.train_data_path)
        # test_seg_data = read_test_data_isic2018_seg(args.val_data_path)
        train_class_data = read_train_data_isic2018_class(data_path)
        test_class_data = read_test_data_isic2018_class(data_path)
    elif args.dataset == "glas":
        # train_seg_data = read_train_data_glas_seg(args.train_data_path)
        # test_seg_data = read_test_data_glas_seg(args.val_data_path)
        train_class_data = read_train_data_glas_class(args.train_data_path)
        test_class_data = read_test_data_glas_class(args.val_data_path)
    elif args.dataset == "kvasir":
        data_path = "/home/dj212/lwc/data/Kvasir"
        train_class_data,test_class_data = read_train_test_data_kvasir_class(data_path)
        # train_seg_data,test_seg_data = read_data_kvasir_seg(args.train_data_path)
    elif args.dataset == "voc2012":
        train_class_data,test_class_data = read_data_voc2012_class(args.train_data_path)
        # train_seg_data,test_seg_data = read_data_voc2012_seg(args.train_data_path)
    else:
        print("dataset error")
        exit(0)

    img_size = 224
    data_transform = {
        "train": A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ToTensorV2()
        ]),
        "val": A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(img_size, img_size),
            A.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ToTensorV2()
        ])
    }

    # train_seg_dataset = DataSetMutliTaskSegment(data=train_seg_data, transform=data_transform["train"])
    # test_seg_dataset = DataSetMutliTaskSegment(data=test_seg_data, transform=data_transform["val"])
    train_class_dataset = DataSetMutliTaskClassify(data=train_class_data, transform=data_transform["train"])
    val_class_dataset = DataSetMutliTaskClassify(data=test_class_data, transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_class_loader = torch.utils.data.DataLoader(train_class_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     pin_memory=True,
                                                     num_workers=nw,
                                                     collate_fn=train_class_dataset.collate_fn)
    val_class_loader = torch.utils.data.DataLoader(val_class_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   pin_memory=True,
                                                   num_workers=nw,
                                                   collate_fn=val_class_dataset.collate_fn)
    if args.model == "hifuse":
        model = HiFuse_Small(num_classes=args.num_classes).to(device)
    elif args.model == "resnet18":
        # 使用torchvision提供的resnet18，并修改最后一层全连接层以适应num_classes
        model = resnet18_torch(pretrained=False)  # 不加载预训练权重
        model.fc = torch.nn.Linear(model.fc.in_features, args.num_classes)  # 修改分类头
        model = model.to(device)
    elif args.model == "vgg19":
        model = vgg19(pretrained=False)  # 不加载预训练权重
        model.classifier[6] = torch.nn.Linear(4096, args.num_classes)  # 替换最后的分类层
        model = model.to(device)
    elif args.model == "vit_b_16":
        model = vit_b_16(pretrained=False, num_classes=args.num_classes)  # 不加载预训练权重，设置分类类别
        model = model.to(device)
    elif args.model == "vit_b_32":
        model = vit_b_32(pretrained=False, num_classes=args.num_classes)  # 不加载预训练权重，设置分类类别
        model = model.to(device)

    if args.RESUME == False:
        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load(args.weights, map_location=device)['state_dict']

            # Delete the weight of the relevant category
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            model.load_state_dict(weights_dict, strict=False)

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # All weights except head are frozen
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    # pg = [p for p in model.parameters() if p.requires_grad]
    pg = get_params_groups(model, weight_decay=args.wd)
    # optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    lr_scheduler_class = create_lr_scheduler(optimizer, len(train_class_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)

    best_acc = 0.
    start_epoch = 0

    if args.RESUME:
        path_checkpoint = f"./model_weight/{task_name}/checkpoint/ckpt_best_{args.epoch_start}.pth"
        print("model continue train")
        checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        lr_scheduler_class.load_state_dict(checkpoint['lr_schedule_class'])
        if args.freeze_layers:
            old_optimizer_state = checkpoint['optimizer']
            # 将旧优化器中匹配的状态复制到新优化器
            optimizer.load_state_dict({
                'state': {k: v for k, v in old_optimizer_state['state'].items() if
                          k in optimizer.state_dict()['state']},
                'param_groups': optimizer.state_dict()['param_groups'],
            })
        else:
            optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(start_epoch + 1, args.epochs + 1):

        # train
        train_loss, train_acc = train_only_class(model=model,
                                               optimizer=optimizer,
                                               data_loader=train_class_loader,
                                               device=device,
                                               epoch=epoch,
                                               lr_scheduler=lr_scheduler_class)

        # validate
        val_acc = evaluate_only_class(model=model,
                                     data_loader=val_class_loader,
                                     device=device,
                                     epoch=epoch)
        tags = ["train_loss", "train_acc", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_acc, epoch)
        tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)

        save_dir = os.path.join("model_weight", task_name)
        if best_acc < val_acc:
            save_checkpoint(os.path.join(save_dir, "best_model.pth"), model, epoch)
            print(f"Saved epoch {epoch} as new best model")
            best_acc = val_acc

        if epoch % 50 == 0:
            print('epoch:', epoch)
            print('learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])

            save_checkpoint(os.path.join(save_dir, "checkpoint", "ckpt_best_%s.pth" % (str(epoch))), model, epoch,
                            optimizer=optimizer, lr_scheduler_class=lr_scheduler_class)

        # add loss, acc and lr into tensorboard
        print("[epoch {}] accuracy: {}".format(epoch, round(val_acc, 3)))

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM" % (total / 1e6))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--epoch_start', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--RESUME', type=bool, default=False)

    parser.add_argument('--dataset', type=str, default="isic2018")
    parser.add_argument('--train_data_path', type=str, default="")
    parser.add_argument('--val_data_path', type=str, default="")
    parser.add_argument('--model', type=str, default="dff")

    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')

    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
    
# python train_cls.py --model resnet18 --dataset kvasir --train_data_path /home/dj212/lwc/data/Kvasir --val_data_path /home/dj212/lwc/data/Kvasir --batch-size 16 --num_classes 8
# python train_cls.py --model vgg19 --dataset isic2018 --train_data_path /home/dj212/lwc/data/ISIC2018 --val_data_path /home/dj212/lwc/data/ISIC2018 --batch-size 12 --num_classes 7
