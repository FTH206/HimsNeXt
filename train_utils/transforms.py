import cv2
import numpy as np
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


def pad_if_smaller(img, size, fill=0):

    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)

        image = F.resize(image, size)

        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class Resize(object):
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, image, target):
        image = F.resize(image, self.target_size)
        target = F.resize(target, self.target_size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target



class ElasticDeformation:

    def __init__(self, alpha=30, sigma=5):
        self.aug = iaa.ElasticTransformation(alpha=alpha, sigma=sigma)

    def __call__(self, image, target):

        img_np = np.array(image)
        mask_np = np.array(target).astype(np.uint8)


        if mask_np.ndim == 2:
            mask_np = np.expand_dims(mask_np, axis=-1)  # (H,W) -> (H,W,1)


        segmap = SegmentationMapsOnImage(mask_np, shape=img_np.shape)


        aug_det = self.aug.to_deterministic()
        img_aug, segmap_aug = aug_det(image=img_np, segmentation_maps=segmap)


        mask_aug = segmap_aug.get_arr().squeeze().astype(np.uint8)  # (H,W,1) -> (H,W)
        return Image.fromarray(img_aug), Image.fromarray(mask_aug)


class MultiScaleTileCrop:


    def __init__(self, tile_sizes=[256, 512, 672], pad_value=0):
        self.tile_sizes = tile_sizes
        self.pad_value = pad_value

    def __call__(self, image, target):
        crop_size = random.choice(self.tile_sizes)


        image = pad_if_smaller(image, crop_size, self.pad_value)
        target = pad_if_smaller(target, crop_size, 0)

        i, j, h, w = T.RandomCrop.get_params(image, (crop_size, crop_size))
        return (
            F.crop(image, i, j, h, w),
            F.crop(target, i, j, h, w)
        )


class LimitedRotation:

    def __init__(self, degrees=(-15, 15)):
        self.degrees = degrees

    def __call__(self, image, target):
        angle = random.uniform(*self.degrees)
        return F.rotate(image, angle), F.rotate(target, angle)


class CLAHEAug:

    def __call__(self, img, target):
        img_np = np.array(img)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        if img_np.ndim == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            img_np[..., 0] = clahe.apply(img_np[..., 0])
            img_np = cv2.cvtColor(img_np, cv2.COLOR_LAB2RGB)
        else:
            img_np = clahe.apply(img_np)
        return Image.fromarray(img_np), target


class DynamicCLAHEAug:

    def __init__(self, clip_scale=0.03, tile_ratio=0.1):
        self.clip_scale = clip_scale
        self.tile_ratio = tile_ratio

    def __call__(self, img, target):
        img_np = np.array(img)
        h, w = img_np.shape[:2]

        tile_size = int(max(h, w) * self.tile_ratio)
        tile_size = 8 if tile_size < 8 else tile_size
        clip_limit = self.clip_scale * max(h, w)

        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_size, tile_size)
        )

        if img_np.ndim == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            img_np[..., 0] = clahe.apply(img_np[..., 0])
            img_np = cv2.cvtColor(img_np, cv2.COLOR_LAB2RGB)
        else:
            img_np = clahe.apply(img_np)

        return Image.fromarray(img_np), target


class RandomApply:


    def __init__(self, transforms, p=0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            for t in self.transforms:
                image, target = t(image, target)
        return image, target


class RandomStainAugmentation:

    def __init__(self, h_range=(0.7, 1.3), e_range=(0.7, 1.3), bg_range=(0.9, 1.1)):
        self.h_range = h_range
        self.e_range = e_range
        self.bg_range = bg_range


        self.he_matrix = np.array([
            [0.65, 0.70, 0.29],
            [0.07, 0.99, 0.11],
            [0.27, 0.57, 0.78]
        ])

    def __call__(self, img, target):
        img_np = np.array(img).astype(np.float32) / 255


        od = -np.log(np.maximum(img_np, 1e-6))


        try:

            stains = np.dot(od.reshape(-1, 3), np.linalg.pinv(self.he_matrix))
            stains = stains.reshape(od.shape)


            h_factor = random.uniform(*self.h_range)
            e_factor = random.uniform(*self.e_range)
            bg_factor = random.uniform(*self.bg_range)

            stains[:, :, 0] *= h_factor
            stains[:, :, 1] *= e_factor
            stains[:, :, 2] *= bg_factor

            recon = np.dot(stains.reshape(-1, 3), self.he_matrix)
            recon = recon.reshape(od.shape)
            recon = np.exp(-recon)
            recon = np.clip(recon * 255, 0, 255).astype(np.uint8)

            return Image.fromarray(recon), target
        except:

            return img, target


if __name__ == '__main__':
    dummy_img = Image.new("RGB", (672, 672), color=255)
    dummy_mask = Image.new("L", (672, 672), color=1)

    deformer = ElasticDeformation(alpha=20)
    img_aug, mask_aug = deformer(dummy_img, dummy_mask)

    print(np.array(mask_aug).max())