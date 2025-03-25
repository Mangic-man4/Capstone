import os

folder = r"C:\Users\Alexander\Documents\School\Capstone\Burger_Images\1_Raw"  # Folder containing raw burger images
#folder = r"C:\Users\Alexander\Documents\School\Capstone\Burger_Images\2_Flip_ready"  # Folder containing flip ready burger images
#folder = r"C:\Users\Alexander\Documents\School\Capstone\Burger_Images\3_Cooked"  # Folder containing cooked burger images
files = os.listdir(folder)

# Allowed image extensions
image_extensions = (".jpg", ".jpeg", ".png")

for index, file in enumerate(sorted(files)):
    if file.lower().endswith(image_extensions):  # Check if file is an image
        ext = os.path.splitext(file)[1]  # Get file extension (.jpg, .png, etc.)
        new_name = f"Raw{index+1}{ext}"  # Raw burger images
        #new_name = f"Flip_ready{index+1}{ext}"  #Flip ready burger images
        #new_name = f"Cooked{index+1}{ext}"  #Cooked burger images

        old_path = os.path.join(folder, file)
        new_path = os.path.join(folder, new_name)
        os.rename(old_path, new_path)

print("Renaming complete!")
