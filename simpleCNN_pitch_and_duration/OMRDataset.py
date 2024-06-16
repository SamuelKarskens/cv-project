import os
import pandas as pd
import numpy as np
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from torchvision import datasets, transforms

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None,mean=None,std=None):
        self.img_labels = pd.read_csv(annotations_file,header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.unique_indices_pitch = []
        self.unique_indices_duration = []
        self.pitch_label_mapping = {}
        self.duration_label_mapping = {}
        self.mean = 0.
        self.std = 0.
        self.create_mappings()

    def __len__(self):
        return len(self.img_labels)
    
    def create_mappings(self):
        self.unique_indices_pitch = np.unique(self.img_labels[1].tolist())
        self.unique_indices_duration = np.unique(self.img_labels[2].tolist())
        self.pitch_label_mapping = {self.unique_indices_pitch[i]: i for i in range(len(self.unique_indices_pitch))}
        self.duration_label_mapping = {self.unique_indices_duration[i]: i for i in range(len(self.unique_indices_duration))}

    def __getitem__(self, idx):
        # use directory where folders with images are located, append the folder name (pitch number) and finally the actual image filename
        if self.img_labels.iloc[idx,1] != " null": #when the folder name is not null
            img_path = os.path.join(self.img_dir,f'{self.img_labels.iloc[idx,1]}', self.img_labels.iloc[idx, 0])
        else:
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path).float() / 255.0 
        label_pitch = self.pitch_label_mapping[self.img_labels.iloc[idx, 1]]
        label_duration = self.duration_label_mapping[self.img_labels.iloc[idx, 2]]
        if self.transform:
            image = self.transform(image)

        # print(image.shape)

        return image, label_pitch, label_duration