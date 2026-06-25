import os
import numpy as np
from PIL import Image

def overlay_mask(image, mask, alpha=0.5):
    if isinstance(image, Image.Image):
        image = np.array(image)

    unique_classes = np.unique(mask)
    unique_classes = unique_classes[unique_classes != 0]

    color_list = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
    ]

    color_map = {}
    for i, class_index in enumerate(unique_classes):
        color_map[class_index] = color_list[i % len(color_list)]


    colored_mask = np.zeros_like(image)
    for class_index, color in color_map.items():
        colored_mask[mask == class_index] = color

    overlayed_image = image.copy()
    for class_index in color_map.keys():
        overlayed_image[mask == class_index] = (
            overlayed_image[mask == class_index] * (1 - alpha) + colored_mask[mask == class_index] * alpha
        ).astype(np.uint8)

    return Image.fromarray(overlayed_image)

def batch_overlay_masks(image_dir, mask_dir, output_dir, alpha=0.5):


    os.makedirs(output_dir, exist_ok=True)


    for image_name in os.listdir(image_dir):
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue


        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path).convert('RGB')


        mask_name = os.path.splitext(image_name)[0] + '.png'
        mask_path = os.path.join(mask_dir, mask_name)
        if not os.path.exists(mask_path):
            print(f"Mask not found for image: {image_name}")
            continue


        mask = np.array(Image.open(mask_path).convert('L'))
        print(f"Unique values in mask for {image_name}:", np.unique(mask))


        overlayed_image = overlay_mask(image, mask, alpha=alpha)


        output_path = os.path.join(output_dir, image_name)
        overlayed_image.save(output_path)
        print(f"Saved overlayed image: {output_path}")

image_dir = '../resized_images_1-3'
mask_dir = '../results/label_1-3/test_results_label_1-3_HimsNeXt'
output_dir = '../results/label_1-3/overlayed_images_GlaS_HimsNeXt'


batch_overlay_masks(image_dir, mask_dir, output_dir, alpha=0.5)