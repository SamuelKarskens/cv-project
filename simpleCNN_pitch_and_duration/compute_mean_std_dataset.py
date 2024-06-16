import os
import numpy as np
from PIL import Image

def load_images(folder_path, target_size=None):
    images = []
    for filename_folder in os.listdir(folder_path):
        if filename_folder == '.DS_Store':
                continue
        for filename in os.listdir(f'{folder_path}/{filename_folder}'):
            if filename == '.DS_Store':
                continue
            if filename.endswith(".jpg") or filename.endswith(".png"):  # check for image files
                img_path = os.path.join(folder_path,filename_folder,filename)
                print(img_path)
                with Image.open(img_path) as img:
                    if target_size:
                        img = img.resize(target_size, Image.Resampling.LANCZOS)
                    images.append(np.array(img))
    return np.array(images)

def compute_mean_std(images):
    # Calculate mean and std dev across all images
    mean = np.mean(images, axis=(0, 1, 2))
    std = np.std(images, axis=(0, 1, 2))
    return mean, std

# Example usage:
folder_path = '../datasets/different_notes/'  # Change this to your image directory path
images = load_images(folder_path, target_size=(150, 150))  # Optional resizing
mean, std = compute_mean_std(images)

print("Mean of the dataset:", mean)
print("Standard deviation of the dataset:", std)
