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

if torch.cuda.is_available():
    print("cuda is available")
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

# torch.cuda.empty_cache()


transform = transforms.Compose([ # composing several transforms together
        # transforms.Resize((350,80)), # resize the image to 256x256
        transforms.ToTensor(), # to tensor object
     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]) # mean = 0.5, std = 0.5

# set batch_size
batch_size = 2

# set number of workers
num_workers = 2

train_set = datasets.ImageFolder('../datasets/notes1', transform=transform)

test_set = datasets.ImageFolder('../datasets/notes1-test', transform=transform)

# print(len(dataset))
# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
# train_size = len(dataset) - test_size
# Todo is it a problem that we dont know if all the classes are present in the testset?
# train_set, test_set = data.random_split(dataset, [train_size, test_size])

train_loader = data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)
test_loader = data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)
#
img, label = train_set[0]
print(img.shape,label)
print("Following classes are there : \n",train_set.classes)
print("dataset classes size", len(train_set.classes))
# print("test set classes", len(test_set.classes))

def display_img(img,label):
    print(f"Label : {train_set.classes[label]}")
    plt.imshow(img.permute(1,2,0))
    plt.show()

#display the first image in the dataset
display_img(*train_set[0])

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
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');
    plt.show()


plot_accuracies(history)

def plot_losses(history):
    """ Plot the losses in each epoch"""
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');
    plt.show()

plot_losses(history)
print("done")