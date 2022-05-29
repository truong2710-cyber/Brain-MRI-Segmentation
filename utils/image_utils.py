import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import sys
#sys.path.insert(0, '..')
#from dataloader import *
import os
from utils.dataloader import MRIDataset
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def tensor_to_image(tensor):
    image = tensor.clone().cpu().numpy()
    image = image.transpose((1, 2, 0))
    image = image * 255
    image = image.astype(int)
    return image

def tensor_to_mask(tensor):
    mask = tensor.clone().detach().cpu().numpy()
    return np.squeeze(mask)

def plot_image(number, loader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    iter_ = iter(loader)
    images, masks = next(iter_)
    images = images.to(device) 
    masks = masks.to(device)
    plt.figure(figsize=(10, 10))
    for i in range(number):
        image = tensor_to_image(images[i])
        plt.subplot(2, number, i+1)
        plt.imshow(image)

    for i in range(number):
        mask = tensor_to_mask(masks[i])
        plt.subplot(2, number, i+1+number)
        plt.imshow(mask)
    plt.show()

def plot_hist(train_loss, val_loss):
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend()
    plt.title('Loss Plot')
    plt.show()

def test():
    dataset = MRIDataset('D:\MRI Segmentation\data\kaggle_3m')
    train, val = random_split(dataset, [3600, 329])
    train_loader = DataLoader(dataset=train, batch_size=10,shuffle=True)
    val_loader = DataLoader(dataset=val, batch_size=10)
    plot_image(5, train_loader)

if __name__ == "__main__":
    test()