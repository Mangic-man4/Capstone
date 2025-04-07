import os

folder = r"C:\Users\alexa\Documents\School docs\Capstone\Burger_videos\Flip"  # Folder containing raw burger images
files = os.listdir(folder)

# Allowed image extensions
image_extensions = (".mp4")

for index, file in enumerate(sorted(files)):
    if file.lower().endswith(image_extensions):  # Check if file is an image
        ext = os.path.splitext(file)[1]  # Get file extension (.jpg, .png, etc.)
        new_name = f"Video{index+1}{ext}"  # Video

        old_path = os.path.join(folder, file)
        new_path = os.path.join(folder, new_name)
        os.rename(old_path, new_path)

print("Renaming complete!")
