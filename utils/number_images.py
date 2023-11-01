import os
from PIL import Image

def count_images_in_folder(folder_path, valid_extensions=('png', 'jpg', 'jpeg', 'gif', 'bmp','tiff')):
    image_count = 0

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(valid_extensions):
                image_count += 1

    return image_count

folder_path = input("Enter the path to the folder: ")  # Prompt the user for the folder path
image_count = count_images_in_folder(folder_path)

print(f'Total images in {folder_path}: {image_count}')
