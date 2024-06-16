#Partly from https://medium.com/thecyphy/train-cnn-model-with-pytorch-21dafb918f48

import torch.nn as nn
import torch.nn.functional as F
import torch
from matplotlib import pyplot as plt

from utils import accuracy

if torch.cuda.is_available():
    print("cuda is available")
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

class ImageClassificationPitchDuration(nn.Module):
    def training_step(self, batch):
        images, labels_pitch, labels_duration = batch
        images = images.to(device)
        labels_pitch = labels_pitch.to(device)
        labels_duration = labels_duration.to(device)

        logits_pitch, logits_duration = self(images)                  # Generate predictions
        loss_pitch = F.cross_entropy(logits_pitch, labels_pitch) # Calculate loss
        loss_duration = F.cross_entropy(logits_duration, labels_duration) # Calculate loss
        return loss_pitch+loss_duration

    def validation_step(self, batch):
        images, labels_pitch, labels_duration = batch
        images = images.to(device)
        labels_pitch = labels_pitch.to(device)
        labels_duration = labels_duration.to(device)

        logits_pitch, logits_duration = self(images)    # Generate predictions for both pitch and duration

        loss_pitch = F.cross_entropy(logits_pitch, labels_pitch) # Calculate loss pitch
        loss_duration = F.cross_entropy(logits_duration, labels_duration) # Calculate loss duration

        acc_pitch_pred, indices_pitch_correct = accuracy(logits_pitch, labels_pitch) # Calculate accuracy
        acc_duration_pred, indices_duration_correct = accuracy(logits_duration, labels_duration) # Calculate accuracy

        if True:
            # plt image that was correclty classified
            if len(indices_pitch_correct[0]) > 0:
                print("index correct pitch image ", indices_pitch_correct[0])
                # plt.imshow(images[indices_pitch_correct[0][0]].cpu().numpy().transpose(1,2,0))
                # plt.show()
                # print("Correctly classified pitch")

        return {
            'val_loss_pitch': loss_pitch.detach(), 
            'val_loss_duration': loss_duration.detach(),
            'val_loss': loss_pitch.detach()+loss_duration.detach(),
            'val_acc_pitch': acc_pitch_pred, 
            'val_acc_duration': acc_duration_pred
        }
    
    """
        'val_loss_pitch': loss_pitch.detach(), 
        'val_loss_duration': loss_duration.detach(),
        'val_loss': loss_pitch.detach()+loss_duration.detach(),
        'val_acc_pitch': acc_pitch_pred, 
        'val_acc_duration': acc_duration_pred
    """

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        batch_losses_pitch = [x['val_loss_pitch'] for x in outputs]
        batch_losses_duration = [x['val_loss_duration'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        epoch_loss_pitch = torch.stack(batch_losses_pitch).mean()   # Combine losses
        epoch_loss_duration = torch.stack(batch_losses_duration).mean()   # Combine losses
        batch_accs_pitch = [x['val_acc_pitch'] for x in outputs]
        batch_accs_duration = [x['val_acc_duration'] for x in outputs]
        epoch_acc_pitch = torch.stack(batch_accs_pitch).mean()      # Combine accuracies
        epoch_acc_duration = torch.stack(batch_accs_duration).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(),'val_loss_pitch': epoch_loss_pitch.item(),'val_loss_duration': epoch_loss_duration.item(), 'val_acc_pitch': epoch_acc_pitch.item(), 'val_acc_duration': epoch_acc_duration.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_loss_pitch: {:.4f}, val_loss_duration: {:.4f}, val_acc_pitch: {:.4f}, val_acc_duration: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_loss_pitch'], result['val_loss_duration'], result['val_acc_pitch'], result['val_acc_duration']))