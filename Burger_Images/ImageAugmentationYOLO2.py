import cv2
import numpy as np
import os
import glob
import albumentations as A
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Define augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.Rotate(limit=10, p=0.5),  # Small rotation to keep labels accurate
    A.RandomGamma(p=0.3),
    A.MotionBlur(blur_limit=5, p=0.2),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Input and output directories
input_folder = "Capstone\Burger_Images\1_Raw"  # Change this to your folder
label_folder = "Capstone\Burger_Images\1_Raw_Labels"  # YOLO label .txt files
output_folder = "Capstone\Burger_Images\1_Raw_Augmented"
output_label_folder = "Capstone\Burger_Images\1_Raw_Augmented_Labels"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_label_folder, exist_ok=True)

# Get list of images
image_paths = glob.glob(os.path.join(input_folder, "*.jpg"))  # Adjust extension if needed

def read_yolo_label(label_path, img_width, img_height):
    """Reads YOLO label and converts it into Albumentations format."""
    with open(label_path, "r") as f:
        lines = f.readlines()
    
    bboxes = []
    class_labels = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:])

        # Convert to pixel format for Albumentations
        x_min = (x_center - width / 2) * img_width
        y_min = (y_center - height / 2) * img_height
        x_max = (x_center + width / 2) * img_width
        y_max = (y_center + height / 2) * img_height

        bboxes.append([x_min, y_min, x_max, y_max])
        class_labels.append(class_id)
    
    return bboxes, class_labels

def write_yolo_label(save_path, bboxes, class_labels, img_width, img_height):
    """Converts Albumentations format back to YOLO and writes to file."""
    with open(save_path, "w") as f:
        for bbox, class_id in zip(bboxes, class_labels):
            x_min, y_min, x_max, y_max = bbox

            # Convert back to YOLO format
            x_center = (x_min + x_max) / 2 / img_width
            y_center = (y_min + y_max) / 2 / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height

            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def augment_and_save(image_path):
    """Performs augmentation while preserving YOLO labels."""
    img = cv2.imread(image_path)
    if img is None:
        return

    img_height, img_width = img.shape[:2]
    filename = os.path.basename(image_path).split('.')[0]
    label_path = os.path.join(label_folder, f"{filename}.txt")

    # Read YOLO label if exists
    if os.path.exists(label_path):
        bboxes, class_labels = read_yolo_label(label_path, img_width, img_height)
    else:
        bboxes, class_labels = [], []

    for i in range(4):  # 4 augmentations per image
        augmented = transform(image=img, bboxes=bboxes, class_labels=class_labels)
        aug_img = augmented["image"]
        aug_bboxes = augmented["bboxes"]
        aug_labels = augmented["class_labels"]

        # Save image
        save_img_path = os.path.join(output_folder, f"{filename}_aug{i}.jpg")
        cv2.imwrite(save_img_path, aug_img)

        # Save corresponding YOLO label
        save_label_path = os.path.join(output_label_folder, f"{filename}_aug{i}.txt")
        write_yolo_label(save_label_path, aug_bboxes, aug_labels, img_width, img_height)

# Use multiprocessing for speed
if __name__ == "__main__":
    with Pool(cpu_count()) as p:
        list(tqdm(p.imap(augment_and_save, image_paths), total=len(image_paths)))

print("Augmentation complete!")
