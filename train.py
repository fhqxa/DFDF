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
from model.dff import DFFV1_Tiny,DFFV1_Small,DFFV1_Base,DFFV1_Mini
from model.dffv2 import DFFV2_Tiny,DFFV2_Small,DFFV2_Base
from model.dffv3 import DFFV3_Tiny,DFFV3_Small,DFFV3_Base
from torchvision.models import resnet18 as resnet18_torch
from utils import create_lr_scheduler, get_params_groups, train_one_epoch, evaluate, save_checkpoint
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    task_name = f"{args.model}_{args.dataset}"
    torch.cuda.empty_cache()
    print(f"using {device} device.")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')

    tb_writer = SummaryWriter(f"runs/{task_name }")

    if args.dataset == "isic2018":
        train_seg_data = read_train_data_isic2018_seg(args.train_data_path)
        test_seg_data = read_test_data_isic2018_seg(args.val_data_path)
        train_class_data = read_train_data_isic2018_class(args.train_data_path)
        test_class_data = read_test_data_isic2018_class(args.val_data_path)
    elif args.dataset == "isic2018_train":
        train_seg_data = read_train_data_isic2018_seg(args.train_data_path)
        test_seg_data = read_test_data_isic2018_seg(args.val_data_path)
        class_data = read_train_data_isic2018_class(args.train_data_path)
        train_class_data = class_data[:int(len(class_data)*0.7)]
        test_class_data = class_data[int(len(class_data)*0.7):]
    elif args.dataset == "glas":
        train_seg_data = read_train_data_glas_seg(args.train_data_path)
        test_seg_data = read_test_data_glas_seg(args.val_data_path)
        train_class_data = read_train_data_glas_class(args.train_data_path)
        test_class_data = read_test_data_glas_class(args.val_data_path)
    elif args.dataset == "kvasir":
        train_class_data,test_class_data = read_train_test_data_kvasir_class(args.train_data_path)
        train_seg_data,test_seg_data = read_data_kvasir_seg(args.train_data_path)
    elif args.dataset == "voc2012":
        train_class_data,test_class_data = read_data_voc2012_class(args.train_data_path)
        train_seg_data,test_seg_data = read_data_voc2012_seg(args.train_data_path)
    elif args.dataset == "isic2019":
        train_images_path, train_images_label = read_train_data_isic2019(args.train_data_path,
                                                                         "ISIC_2019_Training_GroundTruth.csv")
        val_images_path, val_images_label = read_val_data_isic2019(args.val_data_path,
                                                                   "ISIC_2019_Training_GroundTruth.csv")
    else:
        print("dataset error")
        exit(0)

    img_size = 224
    data_transform = {
        "train": A.Compose([
            A.Resize(256, 256),
            A.RandomCrop(img_size, img_size),
            # A.CenterCrop(img_size, img_size),
            # A.RandomRotate90(p=0.5),
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

    train_seg_dataset = DataSetMutliTaskSegment(data=train_seg_data, transform=data_transform["train"])
    test_seg_dataset = DataSetMutliTaskSegment(data=test_seg_data, transform=data_transform["val"])
    train_class_dataset = DataSetMutliTaskClassify(data=train_class_data, transform=data_transform["train"])
    val_class_dataset = DataSetMutliTaskClassify(data=test_class_data, transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_seg_loader = torch.utils.data.DataLoader(train_seg_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=nw,
                                                   collate_fn=train_seg_dataset.collate_fn)
    val_seg_loader = torch.utils.data.DataLoader(test_seg_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=nw,
                                                 collate_fn=test_seg_dataset.collate_fn)
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
    if "dffv1_tiny" in args.model:
        model = DFFV1_Tiny(num_classes=args.num_classes, num_segment=1).to(device)
    elif "dffv1_mini" in args.model:
        model = DFFV1_Mini(num_classes=args.num_classes, num_segment=1).to(device)
    elif "dffv1_small" in args.model:
        model = DFFV1_Small(num_classes=args.num_classes, num_segment=1).to(device)
    elif "dffv1_base" in args.model:
        model = DFFV1_Base(num_classes=args.num_classes, num_segment=1).to(device)
    elif "dffv2_tiny" in args.model:
        model = DFFV2_Tiny(num_classes=args.num_classes, num_segment=1).to(device)
    elif "dffv2_small" in args.model:
        model = DFFV2_Small(num_classes=args.num_classes, num_segment=1).to(device)
    elif "dffv2_base" in args.model:
        model = DFFV2_Base(num_classes=args.num_classes, num_segment=1).to(device)
    elif "dffv3_tiny" in args.model:
        model = DFFV3_Tiny(num_classes=args.num_classes, num_segment=1).to(device)
    elif "dffv3_small" in args.model:
        model = DFFV3_Small(num_classes=args.num_classes, num_segment=1).to(device)
    elif "dffv3_base" in args.model:
        model = DFFV3_Base(num_classes=args.num_classes, num_segment=1).to(device)



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
    lr_scheduler_class = create_lr_scheduler(optimizer, len(train_class_loader)+len(train_seg_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)
    lr_scheduler_seg = create_lr_scheduler(optimizer, len(train_seg_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)

    best_acc = 0.
    start_epoch = 0

    if args.RESUME:
        path_checkpoint = f"./model_weight/checkpoint/ckpt_best_{args.epoch_start}.pth"
        print("model continue train")
        checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        lr_scheduler_seg.load_state_dict(checkpoint['lr_schedule_seg'])
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
        train_seg_loss, train_seg_acc = train_one_epoch(model=model,
                                                    optimizer=optimizer,
                                                    data_loader=train_seg_loader,
                                                    device=device,
                                                    epoch=epoch,
                                                    lr_scheduler=lr_scheduler_class,
                                                    type_="seg")
        train_class_loss, train_class_acc = train_one_epoch(model=model,
                                                        optimizer=optimizer,
                                                        data_loader=train_class_loader,
                                                        device=device,
                                                        epoch=epoch,
                                                        lr_scheduler=lr_scheduler_class,
                                                        type_="class")

        # validate
        val_seg_acc = evaluate(model=model,
                             data_loader=val_seg_loader,
                             device=device,
                             epoch=epoch,
                             type_="seg")
        val_class_acc = evaluate(model=model,
                             data_loader=val_class_loader,
                             device=device,
                             epoch=epoch,
                             type_="class")

        tags = ["train_loss", "train_class_acc", "train_seg_acc", "val_class_acc", "val_seg_acc",
                "learning_rate"]
        tb_writer.add_scalar(tags[0], train_seg_loss + train_class_loss, epoch)
        tb_writer.add_scalar(tags[1], train_class_acc, epoch)
        tb_writer.add_scalar(tags[2], train_seg_acc, epoch)
        tb_writer.add_scalar(tags[3], val_class_acc, epoch)
        tb_writer.add_scalar(tags[4], val_seg_acc, epoch)
        tb_writer.add_scalar(tags[5], optimizer.param_groups[0]["lr"], epoch)

        save_dir = os.path.join("model_weight", task_name)
        if best_acc < val_class_acc:
            save_checkpoint(os.path.join(save_dir, "best_model.pth"), model, epoch)
            print(f"Saved epoch {epoch} as new best model")
            best_acc = val_class_acc

        if epoch % 50 == 0:
            print('epoch:', epoch)
            print('learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])

            save_checkpoint(os.path.join(save_dir, "last_model.pth"),model, epoch,optimizer, lr_scheduler_seg, lr_scheduler_class)

        # add loss, acc and lr into tensorboard
        print("[epoch {}] accuracy: {}".format(epoch, round(val_class_acc, 3)))
        torch.cuda.empty_cache()

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

# python train.py --model dff --dataset Kvasir --train_data_path /home/dj212/lwc/data/Kvasir --val_data_path /home/dj212/lwc/data/Kvasir --batch-size 14 --num_classes 8
# python train.py --model dffv2 --dataset isic2018 --train_data_path /home/dj212/lwc/data/ISIC2018 --val_data_path /home/dj212/lwc/data/ISIC2018 --batch-size 18 --num_classes 7
# python train.py --model dffv1_small_0-150_re --dataset isic2018 --train_data_path /home/dj212/lwc/data/ISIC2018 --val_data_path /home/dj212/lwc/data/ISIC2018 --batch-size 12 --num_classes 7
# python train.py --model dffv1_base_0-100 --dataset isic2018_train --train_data_path /home/dj212/lwc/data/ISIC2018 --val_data_path /home/dj212/lwc/data/ISIC2018 --batch-size 12 --num_classes 7
