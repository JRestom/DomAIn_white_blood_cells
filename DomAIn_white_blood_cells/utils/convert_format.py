import os
from PIL import Image

def convert_images_to_png(input_folder, output_folder):
    valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp','.tiff')

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(valid_extensions):
                relative_path = os.path.relpath(root, input_folder)
                output_subfolder = os.path.join(output_folder, relative_path)

                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)

                input_path = os.path.join(root, file)
                output_path = os.path.join(output_subfolder, file)

                try:
                    img = Image.open(input_path)
                    img.save(output_path.replace(file, os.path.splitext(file)[0] + '.png'))
                    print(f'Converted: {input_path} -> {output_path}')
                except Exception as e:
                    print(f'Error converting {input_path}: {e}')

if __name__ == "__main__":
    input_folder = input("Enter the path to the input folder: ")
    output_folder = input("Enter the path to the output folder for PNG images: ")

    convert_images_to_png(input_folder, output_folder)
