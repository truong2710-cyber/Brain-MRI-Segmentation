import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'model/')
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'metric/')
from model.unet import *
from model.unet_plus_plus import *
from model.backboned_unet import *
from utils.dataloader import *
from utils.image_utils import *
from metric.iou import IoU
from metric.dice_coef import DiceCoefficient

SAVE_PATH = {'unet': 'checkpoints\\unet', 
            'unet_plus_plus': 'checkpoints\\unet_plus_plus',
            'backboned_unet': 'checkpoints\\backboned_unet'}
DATA_PATH = 'data\\kaggle_3m'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def eval(number, model_name='unet', backbone_name='vgg16'):
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

    dataset = MRIDataset(DATA_PATH)
    train, not_train = random_split(dataset, [3143, 786], generator=torch.Generator().manual_seed(0))
    val, test = random_split(not_train, [393, 393], generator=torch.Generator().manual_seed(0))
    test_loader = DataLoader(dataset=test, batch_size=10, shuffle=True)
    model.eval()
    iter_ = iter(test_loader)
    images, true_masks = next(iter_)
    images = images.to(device, dtype = torch.float32) 
    masks = model.forward(images)
    masks = masks.clone().detach().cpu().numpy()
    masks = masks.transpose((0, 2, 3, 1))
  
    thresholded_masks = []
    for i in range(number):
        _, mask = cv2.threshold(masks[i], 0.7, 1, cv2.THRESH_BINARY)
        thresholded_masks.append(mask)

    iou = IoU()
    dice_coef = DiceCoefficient()
    print("Mean IoU:", iou.eval(np.squeeze(true_masks.clone().detach().cpu().numpy()), np.array(thresholded_masks)))
    print("Mean Dice coefficient:", dice_coef.eval(np.squeeze(true_masks.clone().detach().cpu().numpy()), np.array(thresholded_masks)))
    
    plt.figure(figsize=(20, 20))
    # MRI Scans
    for i in range(number):
        image = tensor_to_image(images[i])
        plt.subplot(3, number, i+1)
        plt.imshow(image)
    # Ground truth masks
    for i in range(number):
        mask = tensor_to_mask(true_masks[i])
        plt.subplot(3, number, i+1+number)
        plt.imshow(mask)
    # Predicted masks
    for i in range(number):
        mask = thresholded_masks[i]
        plt.subplot(3, number, i+1+2*number)
        plt.imshow(mask)
    plt.show()

if __name__ == '__main__':
    eval(10, model_name='backboned_unet', backbone_name='resnet18')