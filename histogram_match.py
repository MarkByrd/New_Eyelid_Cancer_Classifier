from skimage import exposure
from PIL import Image
import numpy as np
import os

def histogram_match_dataset(data_root, data_to):
    """
    Preprocess all images in a folder dataset by histogram matching
    each image to the first image in the dataset.

    Args:
        data_root (str): path to root folder containing class subfolders
    """
    # 1. Find the first image to use as reference
    first_image_path = None
    for class_name in os.listdir(data_root):
        class_dir = os.path.join(data_root, class_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                first_image_path = os.path.join(class_dir, fname)
                break
        if first_image_path:
            break

    if first_image_path is None:
        raise ValueError("No images found in dataset.")

    ref_img = np.array(Image.open(first_image_path).convert("RGB"))
    os.makedirs(data_to, exist_ok = True)
    # 2. Loop through all images and perform histogram matching
    for class_name in os.listdir(data_root):
        class_dir = os.path.join(data_root, class_name)
        new_class_dir = os.path.join(data_to, class_name)
        os.makedirs(new_class_dir, exist_ok = True)
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(class_dir, fname)
            new_img_path = os.path.join(new_class_dir, fname)
            img = np.array(Image.open(img_path).convert("RGB"))

            # Histogram match per channel
            matched = np.zeros_like(img)
            for c in range(3):
                matched[..., c] = exposure.match_histograms(img[..., c], ref_img[..., c])

            # Convert back to PIL and overwrite original image
            matched_img = Image.fromarray(matched.astype(np.uint8))
            matched_img.save(new_img_path)
histogram_match_dataset("./Dataset", "./hist_matched_dataset")