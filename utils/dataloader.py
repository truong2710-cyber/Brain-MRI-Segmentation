import os
import random
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tqdm import tqdm
import cv2
import skimage
from torch.utils.data import Dataset
from torchvision.transforms import Compose

#torch.set_printoptions(profile="full")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATA_PATH = 'D:\MRI Segmentation\data\kaggle_3m'

class MRIDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.patients = [file for file in os.listdir(data_path) if file not in ['data.csv', 'README.md']]
        self.images, self.masks = [], []

        for patient in self.patients:
            for file in os.listdir(os.path.join(self.data_path, patient)):
                if 'mask' in file:
                    self.masks.append(os.path.join(self.data_path, patient, file))
                else:
                    self.images.append(os.path.join(self.data_path, patient, file))
        
        self.images.sort()
        self.masks.sort()
    
    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]
        image = plt.imread(image)
        #image = cv2.resize(image, (256, 256))
        image = image / 255
        image = image.transpose((2, 0, 1))  # pytorch image format (C, W, H)
        image = torch.from_numpy(image)

        mask = plt.imread(mask)
        mask = mask / 255
        mask = np.expand_dims(mask, axis=-1)
        mask = mask.transpose((2, 0, 1))
        mask = torch.from_numpy(mask)
        return (image, mask)

    def __len__(self):
        return len(self.images)

def test():
    dataset = MRIDataset(DATA_PATH)
    print(dataset[134][1])

if __name__ == '__main__':
    test()