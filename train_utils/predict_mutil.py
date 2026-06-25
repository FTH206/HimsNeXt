import os
import time

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from train_utils.create_model import *



def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main(args):
    classes = 1  # exclude background
    weights_path = "./save_weights/MoNuSeg/best_model_MoNuSeg_HimsNeXt.pth"
    test_img_dir = "./MoNuSeg/test/images"
    output_dir = "./results/MoNuSeg/test_results_MoNuSeg_HimsNeXt2"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(test_img_dir), f"image dir {test_img_dir} not found."

    os.makedirs(output_dir, exist_ok=True)

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


    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_classes = 2

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

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    dataset_config = DATASET_STATS[args.dataset_version]
    mean = dataset_config["mean"]
    std = dataset_config["std"]


    data_transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


    supported_formats = ('.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif')
    test_image_paths = [os.path.join(test_img_dir, img_name)
                        for img_name in os.listdir(test_img_dir)
                        if img_name.lower().endswith(supported_formats)]

    test_image_paths.sort()


    model.eval()
    with torch.no_grad():

        init_img = torch.zeros((1, 3, 512, 512), device=device)
        model(init_img)

        for img_path in test_image_paths:

            original_img = Image.open(img_path).convert('RGB')


            img = data_transform(original_img)
            img = torch.unsqueeze(img, dim=0).to(device)


            t_start = time_synchronized()
            output = model(img)

            if isinstance(output, list):
                out_tensor = output[0]
            elif isinstance(output, dict):
                out_tensor = output['out']
            else:
                out_tensor = output
            prediction = out_tensor.argmax(1).squeeze(0)
            t_end = time_synchronized()
            print(f"Inference time for {os.path.basename(img_path)}: {t_end - t_start:.4f}s")


            if isinstance(output, list):
                prediction = output[0].argmax(1).squeeze(0)
            elif isinstance(output, dict) and 'out' in output:
                prediction = output['out'].argmax(1).squeeze(0)
            else:
                prediction = output.argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)


            prediction[prediction == 1] = 255
            prediction[prediction == 0] = 0


            mask = Image.fromarray(prediction, mode='L')
            mask_save_path = os.path.join(output_dir, os.path.basename(img_path).replace(".bmp", "_predict.bmp"))
            mask.save(mask_save_path)
            print(f"Mask saved to {mask_save_path}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="predict")

    parser.add_argument('--dataset-version', default="MoNuSeg", choices=["label_1-3", "GlaS", "MoNuSeg"],
                        help='Dataset version for mean and std values')

    parser.add_argument("--model-type", default="HimsNeXt",
                        choices=["Unet", "MedNeXt-M", "HimsNeXt", "MedT", "UnetPP", "UNext",
                                 "TransUnet", "AttUNet", "UNet3plus", "CMUNeXt_l"],
                        help="Model Type")

    parsed_args = parser.parse_args()

    return parsed_args

if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("save_weights"):
        os.mkdir("save_weights")

    main(args)
