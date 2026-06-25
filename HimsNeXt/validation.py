import torch
from torch.utils import data
import os
import numpy as np
from scipy.spatial.distance import directed_hausdorff

from train_utils import evaluate, transforms as T
from my_dataset import MyDataset
from model.HimsNeXt.HimsNeXt import HimsNeXt
from train_utils.create_model import *


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


class DatasetEval:
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


def calculate_metrics(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.flatten()
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    return precision, recall, f1, accuracy



def calculate_hd95_asd(pred_mask, gt_mask, spacing=(1.0, 1.0)):
    from scipy.ndimage import _ni_support
    from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure


    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    if not np.any(pred_mask) or not np.any(gt_mask):
        return 999.0, 999.0

    border_structure = generate_binary_structure(pred_mask.ndim, 1)


    pred_border = pred_mask ^ binary_erosion(pred_mask, structure=border_structure, iterations=1)

    gt_border = gt_mask ^ binary_erosion(gt_mask, structure=border_structure, iterations=1)


    pred_border_coords = np.argwhere(pred_border)
    gt_border_coords = np.argwhere(gt_border)

    if len(pred_border_coords) == 0 or len(gt_border_coords) == 0:
        return 999.0, 999.0

    dt_gt = distance_transform_edt(~gt_border, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_border, sampling=spacing)

    dists_pred_to_gt = dt_gt[pred_border]
    dists_gt_to_pred = dt_pred[gt_border]

    all_dists = np.concatenate([dists_pred_to_gt, dists_gt_to_pred])

    hd95 = np.percentile(all_dists, 95)

    asd = np.mean(all_dists)

    return hd95, asd


def main(args):
    num_classes = 2
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    assert os.path.exists(args.weights), f"weights {args.weights} not found."

    dataset_config = DATASET_STATS[args.dataset_version]
    mean = dataset_config["mean"]
    std = dataset_config["std"]

    fixed_size = args.fixed_size if hasattr(args, 'fixed_size') and args.fixed_size else None
    val_dataset = MyDataset(args.data_path, train=False,
                            transforms=DatasetEval(mean, std, fixed_size=fixed_size))

    num_workers = 8
    val_loader = data.DataLoader(val_dataset,
                                 batch_size=1,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 shuffle=False,
                                 collate_fn=val_dataset.collate_fn)
    print(f"Total samples in validation dataset: {len(val_loader.dataset)}")

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

    pretrain_weights = torch.load(args.weights, map_location='cpu')
    if "model" in pretrain_weights:
        model.load_state_dict(pretrain_weights["model"])
    else:
        model.load_state_dict(pretrain_weights)
    model.to(device)
    model.eval()

    total_hd95 = 0.0
    total_asd = 0.0
    valid_samples = 0

    confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)

    print("Calculating Boundary Metrics (HD95 & ASD)...")
    with torch.no_grad():
        for image, target in val_loader:
            image, target = image.to(device), target.to(device)
            output = model(image)


            if isinstance(output, list):
                output = output[0]
            elif isinstance(output, dict):
                output = output['out']


            pred = output.argmax(dim=1).cpu().numpy().astype(np.uint8)  # [B, H, W] -> [1, H, W]
            target = target.cpu().numpy().astype(np.uint8)  # [1, H, W]


            for i in range(pred.shape[0]):

                p_mask = (pred[i] == 1)
                t_mask = (target[i] == 1)

                h, a = calculate_hd95_asd(p_mask, t_mask)

                if h != 999.0:
                    total_hd95 += h
                    total_asd += a
                    valid_samples += 1

    avg_hd95 = total_hd95 / valid_samples if valid_samples > 0 else 0
    avg_asd = total_asd / valid_samples if valid_samples > 0 else 0
    # ------------------------------------

    val_info = str(confmat)
    print(val_info)
    print(f"Dice Coefficient: {dice:.5f}")

    precision, recall, f1, accuracy = calculate_metrics(confmat.mat)
    print(f"Precision: {precision:.5f}")
    print(f"Recall: {recall:.5f}")
    print(f"F1 Score: {f1:.5f}")
    print(f"Accuracy: {accuracy:.5f}")


    print(f"HD95: {avg_hd95:.5f}")
    print(f"ASD: {avg_asd:.5f}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="validation")

    parser.add_argument("--data-path", default="./", help="root")
    parser.add_argument("--weights", default="./save_weights/label_1-3/best_model_label_1-3_HimsNeXt.pth")
    parser.add_argument("--device", default="cuda:0", help="training device")
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--dataset-version', default="label_1-3", choices=["label_1-3", "GlaS", "MoNuSeg"],
                        help='Dataset version for mean and std values')
    parser.add_argument("--model-type", default="HimsNeXt",
                        choices=["Unet", "MedNeXt-M", "HimsNeXt", "MN-Test", "MedT", "UnetPP", "UNext",
                                 "TransUnet", "AttUNet", "UNet3plus", "CMUNeXt_l"],
                        help="")
    parser.add_argument('--fixed-size', type=int, default=[512,512], nargs='+',
                        help='')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)