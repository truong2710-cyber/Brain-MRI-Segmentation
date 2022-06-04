import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os
import sys
sys.path.insert(0, 'model/')
sys.path.insert(0, 'utils/')
from model.unet import *
from model.unet_plus_plus import *
from model.backboned_unet import *
from utils.dataloader import *
from utils.image_utils import *

SAVE_PATH = {'unet': 'checkpoints\\unet', 
            'unet_plus_plus': 'checkpoints\\unet_plus_plus',
            'backboned_unet': 'checkpoints\\backboned_unet'}
DATA_PATH = 'data\\kaggle_3m'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
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
        inputs = torch.sigmoid(inputs)       
        bce_weight = 0.5
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        loss_final = BCE * bce_weight + dice_loss * (1 - bce_weight)
        return loss_final
    
def train(model_name = 'unet', epochs = 20, backbone_name = 'resnet50'):
    assert model_name in ['unet', 'backboned_unet', 'unet_plus_plus']
    assert backbone_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'vgg16', 'vgg19', 'densenet121', 'densenet161', 'densenet169', 'densenet201']

    dataset = MRIDataset(DATA_PATH)
    train, val = random_split(dataset, [3600, 329], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(dataset=train, batch_size=10, shuffle=True)
    val_loader = DataLoader(dataset=val, batch_size=10)
    
    if model_name == 'unet':
        model = Unet()
    elif model_name == 'unet_plus_plus':
        model = NestedUNet(num_classes=1)
    elif model_name == 'backboned_unet':
        model = BackbonedUnet(backbone_name=backbone_name)
    model.to(device)
    if model_name != 'backboned_unet':
        if len(os.listdir(SAVE_PATH[model_name])) > 0:
            model.load_state_dict(torch.load(os.path.join(SAVE_PATH[model_name], 'best.pth'), map_location=torch.device('cpu')))
    else:
        if len(os.listdir(SAVE_PATH[model_name])) > 0:
            model.load_state_dict(torch.load(os.path.join(SAVE_PATH[model_name], '{}_best.pth'.format(backbone_name)), map_location=torch.device('cpu')))

    criterion = DiceBCELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
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
        
        #torch.save(model.state_dict(), os.path.join(SAVE_PATH[model_name], 'epoch_{}.pth'.format(epoch+1)))
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            if model_name != 'backboned_unet':
                torch.save(model.state_dict(), os.path.join(SAVE_PATH[model_name], 'best.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(SAVE_PATH[model_name], '{}_best.pth').format(backbone_name))
    
    return train_loss, val_loss

if __name__ == '__main__':
    train_loss, val_loss = train(model_name='backboned_unet', epochs=20, backbone_name='vgg16')
    plot_hist(train_loss, val_loss)
