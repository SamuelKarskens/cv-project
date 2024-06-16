import pandas as pd
import os
import numpy as np
from collections import defaultdict

map_durations = {'w': 0, 'h': 1, 'q': 2, 'e': 3, 's': 4}
file_name = "train_data_annotations"
try:
    os.mkdir('handwritten_annotations')
except:
    print('folder already exists')

path = '../datasets/handwritten'

with open(f"handwritten_annotations/{file_name}.csv","w") as file:
    # for pitch  in numbers from -12 to 12,
    # create folder with pitch number,
    for image in os.listdir(f'{path}'):
        # print all characters in the image name except the first one
        # print the pitch number
        # print(image[0])
        # try:
        #     os.mkdir(f'handwritten_annotations/{image[1:].replace(".png", "")}')
        # except:
        #     print('folder already exists')

        file.write(f'{image}, null, {map_durations[image[0]]}\n')
        # print(image[0], image[1:].replace(".png", ""))
        # file.write(f'{image}, {pitch}, {map_durations[image[0]]}\n')

    # for pitch in range(-12, 12):
    #     try:
    #         os.mkdir(f'{path}/{pitch}')
    #     except:
    #         print('folder already exists')
    #         for image in os.listdir(f'{path}}'):
    #             file.write(f'{image}, {pitch}, {map_durations[image[0]]}\n')
    #
    # for folder in os.listdir(path):
    #     for image in os.listdir(f'{path}/{folder}'):
    #         file.write(f'{image}, {folder}, {map_durations[image[0]]}\n')