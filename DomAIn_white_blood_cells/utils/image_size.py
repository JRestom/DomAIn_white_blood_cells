
#Script to get the size of the images

from PIL import Image
import os
import random

def get_image_size(image_path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return width, height
    except Exception as e:
        print(f"Error: {e}")
        return None

def list_image_files(folder):
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'))]
    return image_files

if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing images: ")

    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print("Invalid folder path.")
        sys.exit(1)

    image_files = list_image_files(folder_path)

    if len(image_files) < 10:
        print("Not enough images in the folder to sample 10.")
        sys.exit(1)

    random_sample = random.sample(image_files, 10)

    print("Randomly sampled images and their sizes:")
    for image_file in random_sample:
        image_path = os.path.join(folder_path, image_file)
        size = get_image_size(image_path)
        if size:
            print(f"{image_file}: {size[0]} x {size[1]} pixels")