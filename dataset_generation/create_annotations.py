import pandas as pd
import os
import numpy as np
from collections import defaultdict
map_durations = {'w': 0, 'h': 1, 'q': 2, 'e': 3, 's': 4}
file_name = "train_data_annotations"
try:
    os.mkdir('different_notes_3_annotations')
except:
    print('folder already exists')

path = '../datasets/different_notes_3'

with open(f"different_notes_3_annotations/{file_name}.csv","w") as file:
    for folder in os.listdir(path):
        for image in os.listdir(f'{path}/{folder}'):
            file.write(f'{image}, {folder}, {map_durations[image[0]]}\n')