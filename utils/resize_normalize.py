import os
import cv2

# Specify the list of folder names to consider
allowed_folders = ['bas', 'eos', 'lym', 'neu', 'mon']

def resize_and_normalize_images(input_folder, output_folder, target_size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')

    for root, dirs, files in os.walk(input_folder):
        for folder in dirs[:]:  # Create a copy of dirs to avoid modifying it while iterating
            if folder not in allowed_folders:
                dirs.remove(folder)  # Exclude folders with names not in the allowed list

        for file in files:
            if file.lower().endswith(valid_extensions):
                relative_path = os.path.relpath(root, input_folder)
                output_subfolder = os.path.join(output_folder, relative_path)

                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)

                input_path = os.path.join(root, file)
                output_path = os.path.join(output_subfolder, file)

                try:
                    image = cv2.imread(input_path)
                    resized_image = cv2.resize(image, target_size)
                    
                    # Normalize pixel values to the range [0, 1]
                    normalized_image = resized_image / 255.0
                    
                    cv2.imwrite(output_path, normalized_image * 255)  # Save as integers (0-255)
                    print(f'Resized and normalized: {input_path} -> {output_path}')
                except Exception as e:
                    print(f'Error processing {input_path}: {e}')

if __name__ == "__main__":
    input_folder = input("Enter the path to the input folder: ")
    output_folder = input("Enter the path to the output folder for resized and normalized images: ")

    target_size = (256, 256)

    resize_and_normalize_images(input_folder, output_folder, target_size)
