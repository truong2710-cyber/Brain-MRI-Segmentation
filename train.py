import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os
import sys
sys.path.insert(0, '../model')
#sys.path.insert(0, '../utils')
from model.unet import *
from utils.dataloader import *
from utils.image_utils import *

SAVE_PATH = 'D:\\MRI Segmentation\\checkpoints\\unet'
DATA_PATH = 'D:\\MRI Segmentation\\data\\kaggle_3m'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        bce_weight = 0.5
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        loss_final = BCE * bce_weight + dice_loss * (1 - bce_weight)
        return loss_final
    
def train():
    dataset = MRIDataset(DATA_PATH)
    train, val = random_split(dataset, [3600, 329])
    train_loader = DataLoader(dataset=train, batch_size=10,shuffle=True)
    val_loader = DataLoader(dataset=val, batch_size=10)
    
    model = Unet256()
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(SAVE_PATH, 'best.pth')))

    criterion = DiceBCELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    epochs = 20
    train_loss = []
    val_loss = []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        running_train_loss = []
        for (image, mask) in tqdm(train_loader):
            image = image.to(device, dtype = torch.float32)
            mask = mask.to(device, dtype = torch.float32)
            pred = model.forward(image)
            loss = criterion(pred, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_train_loss.append(loss.item())

        running_val_loss = []
        
        with torch.no_grad():
            for image,mask in val_loader:
                image = image.to(device,dtype=torch.float)
                mask = mask.to(device,dtype=torch.float)                            
                pred_mask = model.forward(image)
                loss = criterion(pred_mask,mask)
                running_val_loss.append(loss.item())
        
        epoch_train_loss = np.mean(running_train_loss) 
        print('Train loss: {}'.format(epoch_train_loss))                       
        train_loss.append(epoch_train_loss)
    
        epoch_val_loss = np.mean(running_val_loss)
        print('Validation loss: {}'.format(epoch_val_loss))                                
        val_loss.append(epoch_val_loss)

        torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'epoch_{}.pth'.format(epoch+1)))
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'best.pth'))
    
    return train_loss, val_loss

def main():
    train()

if __name__ == '__main__':
    main()