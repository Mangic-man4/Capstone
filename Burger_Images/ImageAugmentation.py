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
    A.Rotate(limit=15, p=0.5),
    A.RandomGamma(p=0.3),
    A.MotionBlur(blur_limit=5, p=0.2),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3)
])

# Input and output directories
input_folder = "C:\Users\Alexander\Documents\School\Capstone\Burger_Images\1_Raw"  # Change to your actual folder
output_folder = "C:\Users\Alexander\Documents\School\Capstone\Burger_Images\1_Raw_Augmented"
os.makedirs(output_folder, exist_ok=True)

# Get list of images
image_paths = glob.glob(os.path.join(input_folder, "*"))

def augment_and_save(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return
    
    filename = os.path.basename(image_path).split('.')[0]

    for i in range(5):  # Create 5 variations per image
        augmented = transform(image=img)["image"]
        save_path = os.path.join(output_folder, f"{filename}_aug{i}.jpg")
        cv2.imwrite(save_path, augmented)

# Use multiprocessing for speed
if __name__ == "__main__":
    with Pool(cpu_count()) as p:
        list(tqdm(p.imap(augment_and_save, image_paths), total=len(image_paths)))

print("Augmentation complete!")
