import os

import matplotlib.pyplot as plt # for plotting
import numpy as np # for transformation
import torch # PyTorch package
import torchvision # load datasets
import torchvision.transforms as transforms # transform data
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer
from OMRDataset import CustomImageDataset
from OMRClassification import OMRClassification
from torchvision import datasets
from torch.utils import data
from utils import fit


# transform = transforms.Compose([ # composing several transforms together
#     transforms.Resize((150,150)), # resize the image to 256x256
#     transforms.ToTensor(), # to tensor object
#     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ]) # mean = 0.5, std = 0.5

transform = transforms.Compose([
    transforms.Resize((100, 354)),  # Resize all images to 256x256
])

# set batch_size
batch_size = 64

# set number of workers
num_workers = 0
# dataset = datasets.ImageFolder('audiveris_data', transform=transform)

dataset = CustomImageDataset('../dataset_generation/pitch_and_duration_diff_notes/annotations_diff_notes.csv', img_dir='/Users/adrianseguralorente/cv-project/datasets/different_notes_test_only_h',transform=transform)

test_size = 200
train_size = len(dataset) - test_size
#
train_set, test_set = data.random_split(dataset, [train_size, test_size])
# # train_dataset = CustomDataset('dataset2/info.csv', 'dataset2', transform=transform)
train_loader = data.DataLoader(train_set, batch_size=batch_size,
                               shuffle=True, num_workers=num_workers)
# test_loader = data.DataLoader(test_set, batch_size=batch_size,
#                               shuffle=True, num_workers=num_workers)

# img, label = dataset[0]
data_iter = iter(train_loader)

# Get the first batch
first_batch = next(data_iter)
images, labels = first_batch

the_model = OMRClassification()
the_model.load_state_dict(torch.load("models/model.pth"))

the_model.eval()
# image = transform(img)
with torch.no_grad():
    outputs = the_model(images)
    # print(outputs)
    _, preds = torch.max(outputs, dim=1)
    # _, predicted = torch.max(outputs, 1)

# the_model(dataset[1])
# Convert the predicted index to the corresponding class label
# class_index = predicted.item()
# class_labels = os.listdir('audiveris_data')  # Assuming class labels are folder names
# predicted_class = class_labels[class_index]
#
# print(f'Predicted class: {predicted_class}')
print(preds)
print(labels)