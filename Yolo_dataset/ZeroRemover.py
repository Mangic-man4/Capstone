import os
import re

def remove_leading_zeros(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt") or filename.endswith(".jpg"):
            name, ext = os.path.splitext(filename)

            # Match prefix and leading-zero number
            match = re.match(r"([a-zA-Z]+)(0*)(\d+)$", name)
            if match:
                prefix, zeros, number = match.groups()
                new_name = f"{prefix}{int(number)}{ext}"  # removes leading zeros
                src = os.path.join(folder_path, filename)
                dst = os.path.join(folder_path, new_name)
                print(f"Renaming: {filename} → {new_name}")
                os.rename(src, dst)

# Example usage
folder_path = r"C:\Users\alexa\Documents\School docs\Capstone\Yolo_dataset\Dataset 1\images\val"
remove_leading_zeros(folder_path)
