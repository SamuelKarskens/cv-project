# from https://medium.com/thecyphy/train-cnn-model-with-pytorch-21dafb918f48
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

if torch.cuda.is_available():
    print("cuda is available")
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

"""
    'val_loss_pitch': loss_pitch.detach(), 
    'val_loss_duration': loss_duration.detach(),
    'val_loss': loss_pitch.detach()+loss_duration.detach(),
    'val_acc_pitch': acc_pitch_pred, 
    'val_acc_duration': acc_duration_pred
"""

def fit(epochs, lr, model, train_loader, val_loader, opt_func = torch.optim.AdamW):
    history = []
    optimizer = opt_func(model.parameters(),lr) #todo maybe do with weight_decay=1e-4
    # scheduler = StepLR(optimizer, step_size=6, gamma=0.1)
    for epoch in range(epochs):
        print("start epoch", epoch)
        model.train()
        train_losses = []
        for index, batch in tqdm(enumerate(train_loader)):
            combined_loss = model.training_step(batch)
            train_losses.append(combined_loss)
            combined_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # scheduler.step()  # Adjust learning rate
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)

    return history