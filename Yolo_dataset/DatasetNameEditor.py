import os
import re

def bulk_rename_files(folder_path, rename_map, default_base="Raw", test1_base="Flip"):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt") or filename.endswith(".jpg"):
            name, ext = os.path.splitext(filename)

            # Handle special 'test1' case
            if name.startswith("test1"):
                number = name[len("test1"):]
               # number = number.lstrip("0").zfill(3)  # normalize to 3-digit
                number = number.lstrip("0") or "0" 
                new_name = f"{test1_base}{number}{ext}"

            # Handle case where base is empty (just numbers)
            elif re.fullmatch(r"\d+", name):
                #number = name.zfill(3)
                number = name.lstrip("0") or "0"
                new_name = f"{default_base}{number}{ext}"

            else:
                # Generic case: match base and number
                match = re.match(r"([a-zA-Z]+)(\d+)", name)
                if match:
                    base, number = match.groups()
                    if base in rename_map:
                        new_base = rename_map[base]
                        new_name = f"{new_base}{number}{ext}"
                    else:
                        continue  # skip if base not in map
                else:
                    continue  # skip unrecognized format

            src = os.path.join(folder_path, filename)
            dst = os.path.join(folder_path, new_name)
            print(f"Renaming: {filename} → {new_name}")
            os.rename(src, dst)

# Example usage
rename_map = {
    "test1":"Flip"
}

folder_path = r"C:\Users\alexa\Documents\School docs\Capstone\Yolo_dataset\Dataset 1\labels\val"
bulk_rename_files(folder_path, rename_map)
