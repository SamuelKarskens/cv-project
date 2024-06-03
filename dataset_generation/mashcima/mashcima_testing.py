import os

import mashcima as mc
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from mashcima import CanvasOptions

HIGHEST_PITCH = 12  # fourth ledger line above
LOWEST_PITCH = -12  # fourth ledger line below

PITCHES = [str(pos) for pos in reversed(range(LOWEST_PITCH, HIGHEST_PITCH + 1))]
lttr = PITCHES
dataset_path = "../../datasets/notes1/"
names = []
for _ in range(5):
    for l in lttr:
        a = f'h{l}'
        #create folder with name from pitch value if not exist yet
        dataset_path_folder = dataset_path + a
        print(f'Creating folder {dataset_path_folder}')
        #create folder
        try:
            os.makedirs(dataset_path_folder)
        except FileExistsError:
            print(f'Folder {dataset_path_folder} already exists')


        canvas_options = CanvasOptions.get_empty()
        canvas_options.random_space_probability = 0.0
        canvas_options.randomize_stem_flips_for_pitches = []

        img_ori = mc.synthesize(
            a,
            # remove random spaces
            main_canvas_options=canvas_options,
            above_canvas_options=canvas_options,
            below_canvas_options=canvas_options,

            transform_image=False, # do not transform
            min_width=100)
        # img_ori = mc.synthesize_for_beauty(a, )
        img = img_ori.copy()
        img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
        img_normalized = img_normalized.astype(np.uint8)
        pil_img = Image.fromarray(img_normalized)
        pil_img = pil_img.convert('RGB')
        # add padding white pixels to get images of 150 x 150
        #with only white pixels around the main image to get 150 x 150
        # pil_img = pil_img.resize((150, 150), Image.ANTIALIAS)

        # Accessing pixel data
        pixels = pil_img.load()
        width, height = pil_img.size
        # Example: Inverting colors
        for i in range(width):
            for j in range(height):
                r, g, b = pixels[i, j]
                pixels[i, j] = (255 - r, 255 - g, 255 - b)
        img = np.array(pil_img)
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.close()
        file_name = f'{dataset_path_folder}/h{l}-0.png'
        if file_name not in names:
            print(f'{file_name} is not in the array, apppending it')
            names.append(file_name)
        else:
            print(f'{file_name} in the array')
            counter = 1
            while file_name in names:
                file_name = f'{dataset_path_folder}/h{l}-{counter}.png'
                print(f'Checking {file_name}')
                counter += 1
            names.append(file_name)
        pil_img.save(file_name)