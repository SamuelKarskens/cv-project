#code partly from tutorial online

import matplotlib.pyplot as plt # for plotting
import numpy as np # for transformation
import torch # PyTorch package
import torchvision # load datasets
import torchvision.transforms as transforms # transform data
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer
from OMRClassification import OMRClassification
from torchvision import datasets
from torch.utils import data
from utils import fit
from OMRDataset import CustomImageDataset
from torchviz import make_dot


if torch.cuda.is_available():
    print("cuda is available")
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

# torch.cuda.empty_cache()


# transform = transforms.Compose([ # composing several transforms together
#         # transforms.Resize((350,80)), # resize the image to 256x256
#         transforms.ToTensor(), # to tensor object
#      # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ]) # mean = 0.5, std = 0.5

transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.Grayscale(),
])
transform_training = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.Grayscale(),
    transforms.RandomRotation(5, fill=1),
])
# set batch_size
batch_size = 64

# set number of workers
num_workers = 0

# train_set = datasets.ImageFolder('../datasets/different_notes', transform=transform)

# test_set = datasets.ImageFolder('../datasets/different_notes_test_only_h', transform=transform)

# train_set = CustomImageDataset('../dataset_generation/different_notes_3_annotations/train_data_annotations.csv', img_dir='../datasets/different_notes_3',transform=transform_training)

train_set = CustomImageDataset('../dataset_generation/pitch_and_duration_diff_notes/train_data_annotations.csv', img_dir='../datasets/different_notes',transform=transform_training)
# train_set = CustomImageDataset('../dataset_generation/pitch_and_duration_diff_notes_verovio/train_data_annotations_verovio.csv', img_dir='../datasets/data_verovio',transform=transform_training)

test_set = CustomImageDataset('../dataset_generation/handwritten_annotations/train_data_annotations.csv', img_dir='../datasets/handwritten',transform=transform)

# test_set = CustomImageDataset('../dataset_generation/pitch_and_duration_diff_notes/train_data_annotations.csv', img_dir='../datasets/different_notes',transform=transform)
# test_set = CustomImageDataset('../dataset_generation/pitch_and_duration_diff_notes/annotations_diff_notes.csv', img_dir='../datasets/different_notes_test_only_h',transform=transform)

train_loader = data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)
test_loader = data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)

#

# compoute mean and stdev of training data
# loader = torch.utils.data.DataLoader(train_set, batch_size=2, num_workers=0, shuffle=False)
# mean = 0.
# std = 0.
# nb_samples = 0.
# for data, _, _ in loader:
#     batch_samples = data.size(0)  # batch size (the last batch can have smaller size!)
#     data = data.view(batch_samples, data.size(1), -1)
#     mean += data.mean(2).sum(0)
#     std += data.std(2).sum(0)
#     nb_samples += batch_samples

# mean /= nb_samples
# std /= nb_samples

# train_set.transform = transform = transforms.Compose([
#     # transforms.Resize((256, 256)),  # Resize all images to 256x256
#     transforms.Normalize(mean=mean, std=std)  # Normalization parameters
# ])

# print(train_set.transform)

# print(mean)
# print(std)
# exit()


img, label_pitch, label_duration = train_set[0]
print(img.shape,label_pitch)
# print("Following classes are there : \n",train_set.classes)
# print("dataset classes size", len(train_set.classes))
# print("test set classes", len(test_set.classes))

# def display_img(img,label, duration):
#     # print(f"Label : {train_set.classes[label]}")
#     plt.imshow(img.permute(1,2,0))
#     plt.show()
# #
# # #display the first image in the dataset
# display_img(*train_set[0])

# def display_img(img,label):
#     print(f"Label : {dataset.classes[label]}")
#     plt.imshow(img.permute(1,2,0))
#
# #display the first image in the dataset
# display_img(*dataset[0])

num_epochs = 30
opt_func = torch.optim.Adam
lr = 0.001
model = OMRClassification()

model.to(device)

#fitting the model on training data and record the result after each epoch
history = fit(num_epochs, lr, model, train_loader, test_loader, opt_func)
torch.save(model.state_dict(), "models/model.pth")

# the_model = OMRClassification()
# the_model.load_state_dict(torch.load("models/model.pth"))
def plot_accuracies(history):
    """ Plot the history of accuracies"""
    accuracies_duration = [x['val_acc_duration'] for x in history]
    accuracies_pitch = [x['val_acc_pitch'] for x in history]
    #plot accuracies_duration clear to see which is it

    plt.plot(accuracies_duration, '-bx')
    plt.plot(accuracies_pitch, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Duration', 'Pitch'])
    plt.title('Accuracy vs. No. of epochs')
    plt.show()


plot_accuracies(history)

def plot_losses(history):
    """ Plot the losses in each epoch"""
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    val_loss_pitch = [x['val_loss_pitch'] for x in history]
    val_loss_duration = [x['val_loss_duration'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.plot(val_loss_pitch, '-yx')
    plt.plot(val_loss_duration, '-gx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation', 'Validation pitch', 'Validation duration'])
    plt.title('Loss vs. No. of epochs')
    plt.show()

plot_losses(history)
print("done")