import os
import time
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
from train_utils import train_one_epoch, evaluate, create_lr_scheduler, transforms as T
from my_dataset import MyDataset
from train_utils.visualizer import TrainingVisualizer
from train_utils.create_model import *

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def set_seed(seed=42):

    random.seed(seed)

    if not isinstance(seed, int):
        seed = int(seed)
    seed = seed % (2**32)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"The random seed has been set to: {seed}")


DATASET_STATS = {
    "label_1-3": {
        "mean": (0.680, 0.505, 0.775),
        "std": (0.170, 0.192, 0.094)
    },
    "GlaS": {
        "mean": (0.787, 0.511, 0.785),
        "std": (0.157, 0.213, 0.116)
    },
    "MoNuSeg": {
        "mean": (0.644, 0.447, 0.604),
        "std": (0.189, 0.192, 0.153)
    }
}


class SegmentationPresetTrain:
    def __init__(self, base_size, tile_sizes, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.669, 0.522, 0.778), std=(0.163, 0.181, 0.089),
                 fixed_size=None):

        trans = []
        

        if fixed_size is not None:
            trans.append(T.Resize(fixed_size))
        else:
            min_size = int(0.5 * base_size)
            max_size = int(1.2 * base_size)
            trans.append(T.RandomResize(min_size, max_size))
            
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))

        if fixed_size is None:
            trans.append(T.MultiScaleTileCrop(tile_sizes))
        
        trans.extend([
            T.RandomApply([T.LimitedRotation()], p=0.5),
            T.RandomApply([T.ElasticDeformation(alpha=20)], p=0.5),
            T.DynamicCLAHEAug(clip_scale=0.03, tile_ratio=0.03),
            T.RandomApply([T.RandomStainAugmentation(h_range=(0.8, 1.2), e_range=(0.8, 1.2))], p=0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        img, target = self.transforms(img, target)
        return img, target


class SegmentationPresetEval:
    def __init__(self, mean, std, fixed_size=None):
        trans = []
        if fixed_size is not None:
            trans.append(T.Resize(fixed_size))
            
        trans.extend([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean, std, dataset_version="GlaS", fixed_size=None):

    if dataset_version == "GlaS":
        base_size = 512
        tile_sizes = [256, 384, 512]
    elif dataset_version == "MoNuSeg":
        base_size = 1000
        tile_sizes = [256, 512, 672]
    else:
        base_size = 672
        tile_sizes = [224, 336, 672]

    if train:
        return SegmentationPresetTrain(base_size, tile_sizes, mean=mean, std=std, fixed_size=fixed_size)
    else:
        return SegmentationPresetEval(mean=mean, std=std, fixed_size=fixed_size)


def main(args):

    if args.use_seed:
        set_seed(args.seed)
        print(f"Use a fixed random seed: {args.seed}")
    elif args.resume:

        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'seed' in checkpoint:
            seed = checkpoint['seed']
            set_seed(seed)
            print(f"Restore random seed from checkpoint: {seed}")
        else:
            seed = torch.initial_seed()
            print(f"There is no random seed information in the checkpoint, using the current random seed: {seed}")
    else:

        print("Without using a fixed random seed, the training results may vary each time")
        
        seed = torch.initial_seed()
        print(f"Current PyTorch random seed: {seed}")
        
        results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        with open(results_file, "a") as f:
            f.write(f"Fixed random seed not used, current PyTorch random seed: {seed}\n\n")
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    # segmentation num_classes + background
    num_classes = args.num_classes + 1

    dataset_config = DATASET_STATS[args.dataset_version]
    mean = dataset_config["mean"]
    std = dataset_config["std"]

    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    fixed_size = args.fixed_size if hasattr(args, 'fixed_size') and args.fixed_size else None
    
    train_dataset = MyDataset(args.data_path,
                             train=True,
                             transforms=get_transform(train=True, mean=mean, std=std, 
                                                     dataset_version=args.dataset_version,
                                                     fixed_size=fixed_size))

    val_dataset = MyDataset(args.data_path,
                           train=False,
                           transforms=get_transform(train=False, mean=mean, std=std, 
                                                   dataset_version=args.dataset_version,
                                                   fixed_size=fixed_size))

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    if args.model_type == "Unet":
        model = create_Unet(num_classes)
    elif args.model_type == "MedNeXt-M":
        model = create_mednextv1_medium(num_classes)
    elif args.model_type == "HimsNeXt":
        model = create_HimsNeXt(num_classes)
    elif args.model_type == "MN-Test":
        model = create_test(num_classes)
    elif args.model_type == "MedT":
        model = create_MedT(num_classes)
    elif args.model_type == "UnetPP":
        model = create_UnetPP(num_classes)
    elif args.model_type == "UNext":
        model = create_UNeXt(num_classes)
    elif args.model_type == "TransUnet":
        model = create_TransUnet(num_classes)
    elif args.model_type == "AttUNet":
        model = create_AttUNet(num_classes)
    elif args.model_type == "UNet3plus":
        model = create_UNet3plus(num_classes)
    elif args.model_type == "CMUNeXt_l":
        model = create_CMUNeXt_l(num_classes)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    model.to(device)

    print(model)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(
        params_to_optimize,
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    lr_scheduler = create_lr_scheduler(
        optimizer, 
        len(train_loader), 
        args.epochs, 
        warmup=True
    )

    checkpoint_dir = os.path.join("save_weights", "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    best_dice = 0.

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if 'best_dice' in checkpoint:
            best_dice = checkpoint['best_dice']
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])
        print(f"Resume training from round {args.start_epoch-1}, current best Dice{best_dice:.5f}")


    visualizer = TrainingVisualizer()
    

    if args.resume:
        visualizer.load_metrics(args.start_epoch - 1)


    start_time = time.time()


    for epoch in range(args.start_epoch, args.epochs):


        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler,)


        confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        print(f"dice coefficient: {dice:.5f}")

        visualizer.update_metrics(epoch, mean_loss, dice, lr)


        with open(results_file, "a") as f:

            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"dice coefficient: {dice:.5f}\n"

            f.write(train_info + val_info + "\n\n")


        latest_checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "best_dice": best_dice,
            "args": args,
            "seed": torch.initial_seed()
        }
        if args.amp:
            latest_checkpoint["scaler"] = scaler.state_dict()
        
        latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
        try:
            torch.save(latest_checkpoint, latest_checkpoint_path)
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")

            try:

                for old_file in os.listdir(checkpoint_dir):
                    if old_file.startswith("checkpoint_epoch"):
                        os.remove(os.path.join(checkpoint_dir, old_file))
                        print(f"The old checkpoint file has been deleted: {old_file}")
                torch.save(latest_checkpoint, latest_checkpoint_path)
                print(f"After cleaning, the checkpoint was successfully saved to {latest_checkpoint_path}")
            except Exception as e2:
                print(f"The attempt to save the checkpoint failed again: {e2}")


        for i in range(epoch):
            previous_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{i}.pth")
            if os.path.exists(previous_checkpoint_path):
                try:
                    os.remove(previous_checkpoint_path)
                except:
                    pass


        if args.save_best is True:

            if best_dice < dice:
                best_dice = dice

                save_file = {"model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "epoch": epoch,
                            "best_dice": best_dice,
                            "args": args,
                            "seed": torch.initial_seed()
                            }
                if args.amp:
                    save_file["scaler"] = scaler.state_dict()

                torch.save(save_file, "save_weights/best_model_GlaS_UNet.pth")
                print(f"The best model has been saved, Dice: {best_dice:.3f}")
            else:
                pass


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))


    print("training time {}".format(total_time_str))
    

    print(f"Best Dice coefficient: {best_dice:.4f}")
    

    with open(results_file, "a") as f:
        f.write(f"\nTraining completed\nTotal training time: {total_time_str}\nBest Dice coefficient: {best_dice:.4f}\n")

    visualizer.plot_final_metrics()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")
    parser.add_argument("--data-path", default="./", help="DATA root")
    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda:2", help="training device")
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=400, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument("--seed", default=42, type=int, help="random seed for reproducibility")
    parser.add_argument("--use-seed", action="store_true", default=True,
                        help="Whether to use a fixed random seed, default is used")
    
    parser.add_argument('--lr', default=8e-4, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    # python train.py --epochs 400 --resume save_weights/checkpoints/latest_checkpoint.pth
    parser.add_argument('--resume', default='', help='resume from checkpoint: "" for no resume')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument('--dataset-version', default="MoNuSeg", choices=["1", "1-2", "label_1-3", "GlaS", "MoNuSeg"],
                        help='Dataset version for mean and std values')
    parser.add_argument("--model-type", default="HimsNeXt", choices=["Unet","MedNeXt-M","HimsNeXt","MN-Test","MedT","UnetPP","UNext",
                                                                                "TransUnet","AttUNet","UNet3plus","CMUNeXt_l"],
                        help="Model Type")
    # 512 or None
    parser.add_argument('--fixed-size', type=int, default=[512,512],
                        help='Fixed image size, such as 224 indicating adjustment to 224x224, defaults to None indicating no fixed size')

    parsed_args = parser.parse_args()

    return parsed_args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("save_weights"):
        os.mkdir("save_weights")

    main(args)
