# HimsNeXt

HimsNeXt is a PyTorch project for binary medical image segmentation. The
repository contains the proposed HimsNeXt model together with several baseline
segmentation architectures and shared training, validation, augmentation, and
visualization utilities.

## Supported Models

- HimsNeXt
- U-Net
- MedNeXt-M
- MedT
- U-Net++
- UNeXt
- TransUNet
- Attention U-Net
- UNet 3+
- CMUNeXt-L

## Repository Structure

```text
.
|-- model/                  # Segmentation model implementations
|-- train_utils/            # Training loops, transforms, metrics, visualization
|-- my_dataset.py           # Dataset loader
|-- train.py                # Training entry point
|-- validation.py           # Validation and metric evaluation
|-- requirements.txt        # Python dependencies
`-- .gitignore
```

Dataset folders, trained weights, checkpoints, logs, and experiment outputs are
not committed by default.

## Installation

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate himsnext
```

Alternatively, install the pip dependencies manually:

```bash
pip install -r requirements.txt
```

This project was developed with PyTorch and CUDA. Check the versions in
`requirements.txt` and adjust the PyTorch, CUDA, and MMCV packages to match your
local GPU driver and CUDA toolkit if needed.

## Dataset Layout

The dataset loader expects image files and mask files to be arranged like this:

```text
<data-root>/
`-- MoNuSeg/
    |-- training/
    |   |-- images/
    |   `-- manual/
    `-- test/
        |-- images/
        `-- manual/
```

Supported image formats include `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, and
`.tiff`. Mask files can use common suffixes such as `_manual`, `_anno`, or
`_mask`.

The `--dataset-version` argument selects the normalization statistics used by
the transforms. The current `MyDataset` implementation reads from the `MoNuSeg`
folder under `--data-path`; update `my_dataset.py` if you want to switch the
folder name automatically for other datasets.

## Dataset Sources

- MoNuSeg: https://monuseg.grand-challenge.org/Data/
- Breast Cancer Dataset: https://ieee-dataport.org//documents/breast-cancer-dataset
- GlaS: https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/

## Training

Run training with:

```bash
python train.py \
  --data-path ./ \
  --dataset-version MoNuSeg \
  --model-type HimsNeXt \
  --device cuda:0 \
  --batch-size 4 \
  --epochs 400
```

Useful options:

- `--model-type`: one of `Unet`, `MedNeXt-M`, `HimsNeXt`, `MN-Test`, `MedT`,
  `UnetPP`, `UNext`, `TransUnet`, `AttUNet`, `UNet3plus`, or `CMUNeXt_l`.
- `--dataset-version`: one of `label_1-3`, `GlaS`, or `MoNuSeg`.
- `--resume`: path to a checkpoint, for example
  `save_weights/checkpoints/latest_checkpoint.pth`.
- `--amp`: enable mixed precision training when supported.
- `--use-seed` and `--seed`: control reproducibility.

Training writes checkpoints under `save_weights/checkpoints/` and records
metrics in timestamped `results*.txt` files.

## Validation

Evaluate a trained checkpoint with:

```bash
python validation.py \
  --data-path ./ \
  --weights save_weights/checkpoints/latest_checkpoint.pth \
  --dataset-version MoNuSeg \
  --model-type HimsNeXt \
  --device cuda:0
```

The validation script reports Dice, precision, recall, F1 score, accuracy,
HD95, and ASD.

## Reproducibility

`train.py` sets a fixed random seed by default. When resuming from a checkpoint,
the saved seed is restored when available.

## Notes

- Large datasets and model artifacts are excluded from Git by `.gitignore`.
- Generated caches such as `__pycache__`, `.pytest_cache`, IDE metadata, logs,
  checkpoints, and weight files should not be committed.
- If you publish trained weights separately, include the download link and the
  exact command used to reproduce the reported results.
