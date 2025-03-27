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
    A.GaussNoise(std_range=(0.1, 0.2), p=0.3)  # Adjust range if needed
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.3))

# Input and output directories
#input_folder = "\Capstone\Burger_Images\1_Raw"  # Change this to your folder
#label_folder = "\Capstone\Burger_Images\1_Raw_Labels"  # YOLO label .txt files
#output_folder = "\Capstone\Burger_Images\1_Raw_Augmented"
#output_label_folder = "\Capstone\Burger_Images\1_Raw_Augmented_Labels"

#os.makedirs(output_folder, exist_ok=True)
#os.makedirs(output_label_folder, exist_ok=True)

# Get list of images
#image_paths = glob.glob(os.path.join(input_folder, "*.jpg"))  # Adjust extension if needed

# Get the base directory where the script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Set the paths relative to the script's location
input_folder = os.path.join(base_dir, "1_Raw")  
label_folder = os.path.join(base_dir, "1_Raw_Labels")  
output_folder = os.path.join(base_dir, "1_Raw_Augmented")  
output_label_folder = os.path.join(base_dir, "1_Raw_Augmented_Labels")  

# Create output directories if they don't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_label_folder, exist_ok=True)

# Get list of images
image_paths = []
for ext in ["*.jpg", "*.jpeg", "*.png"]:
    image_paths.extend(glob.glob(os.path.join(input_folder, ext)))

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

        # Convert YOLO format to Albumentations format (normalized)
        x_min = max(0, min(1, x_center - width / 2))
        y_min = max(0, min(1, y_center - height / 2))
        x_max = max(0, min(1, x_center + width / 2))
        y_max = max(0, min(1, y_center + height / 2))

        # Ensure bounding boxes remain valid
        if x_max > x_min and y_max > y_min:
            bboxes.append([x_min, y_min, x_max, y_max])
            class_labels.append(class_id)

    return bboxes, class_labels


def write_yolo_label(save_path, bboxes, class_labels):
    """Converts Albumentations format back to YOLO and writes to file."""
    with open(save_path, "w") as f:
        for bbox, class_id in zip(bboxes, class_labels):
            x_min, y_min, x_max, y_max = bbox

            # Convert back to YOLO format and clamp values
            x_center = max(0, min(1, (x_min + x_max) / 2))
            y_center = max(0, min(1, (y_min + y_max) / 2))
            width = max(0, min(1, x_max - x_min))
            height = max(0, min(1, y_max - y_min))

            # Ensure values are valid before writing
            if width > 0 and height > 0:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


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

    for i in range(4):  # n augmentations per image
        augmented = transform(image=img, bboxes=bboxes, class_labels=class_labels)
        aug_img = augmented["image"]
        aug_bboxes = augmented["bboxes"]
        aug_labels = augmented["class_labels"]

        # Save image
        save_img_path = os.path.join(output_folder, f"{filename}_aug{i}.jpg")
        cv2.imwrite(save_img_path, aug_img)

        # Save corresponding YOLO label
        save_label_path = os.path.join(output_label_folder, f"{filename}_aug{i}.txt")
        write_yolo_label(save_label_path, aug_bboxes, aug_labels)

# Use multiprocessing for speed
if __name__ == "__main__":
    with Pool(cpu_count()) as p:
        list(tqdm(p.imap(augment_and_save, image_paths), total=len(image_paths)))

print("Augmentation complete!")
