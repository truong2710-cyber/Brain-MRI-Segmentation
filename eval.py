import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
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

SAVE_PATH = {'unet': os.path.join('checkpoints', 'unet'),
            'unet_plus_plus': os.path.join('checkpoints', 'unet_plus_plus'),
            'backboned_unet': os.path.join('checkpoints', 'backboned_unet')}
DATA_PATH = os.path.join('data', 'kaggle_3m')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def plot_images(images, true_masks, thresholded_masks, number):
    fig, axes = plt.subplots(3, number, figsize=(number * 2, 6), constrained_layout=True)

    for i in range(number):
        # MRI Scans
        axes[0, i].imshow(tensor_to_image(images[i]))
        axes[0, i].axis('off')

        # Ground truth masks
        axes[1, i].imshow(tensor_to_image(images[i]))
        axes[1, i].imshow(tensor_to_mask(true_masks[i]), cmap='jet', alpha=0.5)
        axes[1, i].axis('off')

        # Predicted masks
        axes[2, i].imshow(tensor_to_image(images[i]))
        axes[2, i].imshow(thresholded_masks[i], cmap='jet', alpha=0.5)
        axes[2, i].axis('off')

    plt.savefig('eval.png', bbox_inches='tight')
    plt.show()


def visualize(args):
    model_name = args.model
    backbone_name = args.backbone
    threshold = args.threshold

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
    test_loader = DataLoader(dataset=test, batch_size=args.num, shuffle=True)
    model.eval()
    iter_ = iter(test_loader)
    images, true_masks = next(iter_)
    images = images.to(device, dtype = torch.float32) 
    breakpoint()
    masks = model.forward(images)
    masks = masks.clone().detach().cpu().numpy()
    masks = masks.transpose((0, 2, 3, 1))
  
    thresholded_masks = []
    for i in range(args.num):
        masks[i] = sigmoid(masks[i])
        _, mask = cv2.threshold(masks[i], threshold, 1, cv2.THRESH_BINARY)
        thresholded_masks.append(mask)

    iou = IoU()
    dice_coef = DiceCoefficient()
    print("Mean IoU:", iou.eval(np.squeeze(true_masks.clone().detach().cpu().numpy()), np.array(thresholded_masks)))
    print("Mean Dice coefficient:", dice_coef.eval(np.squeeze(true_masks.clone().detach().cpu().numpy()), np.array(thresholded_masks)))
    
    plot_images(images, true_masks, thresholded_masks, args.num)

def test(args):
    model_name = args.model
    backbone_name = args.backbone
    batch_size=1
    threshold = args.threshold

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
    test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False)
    model.eval()
    thresholded_masks = []
    true_masks = []
    for (images, true_masks_batch) in tqdm(test_loader):
        images = images.to(device, dtype = torch.float32) 
        masks = model.forward(images)
        masks = masks.clone().detach().cpu().numpy()
        masks = masks.transpose((0, 2, 3, 1))
        true_masks_batch = np.squeeze(true_masks_batch.clone().detach().cpu().numpy())
        true_masks.append(true_masks_batch)
        for i in range(batch_size):
            masks[i] = sigmoid(masks[i])
            _, mask = cv2.threshold(masks[i], threshold, 1, cv2.THRESH_BINARY)
            thresholded_masks.append(mask)
    thresholded_masks = np.array(thresholded_masks)
    true_masks = np.array(true_masks).reshape((-1, 256, 256))
    
    iou = IoU()
    dice_coef = DiceCoefficient()
    if model_name != 'backboned_unet':
        print(model_name)
    else:
        print(model_name, backbone_name)
    print("Mean IoU:", iou.eval(true_masks, thresholded_masks))
    print("Mean Dice coefficient:", dice_coef.eval(true_masks, thresholded_masks))
    

def get_args():
    parser = argparse.ArgumentParser(description="Argument parser for model configuration")

    # Integer argument
    parser.add_argument("--num", type=int, default=3, help="Number of images for visualization")

    # Choice argument (test or vis)
    parser.add_argument("--opt", choices=["test", "vis"], required=True, help="Option mode: test or visualization")

    # Backbone choices
    parser.add_argument("--backbone", choices=["resnet18", "resnet34", "densenet121", "densenet169", "vgg16", "vgg19"], default='densenet121', help="Choose the backbone model")

    # Model type choices
    parser.add_argument("--model", type=str, choices=["backboned_unet", "unet", "unet_plus_plus"], default='backboned_unet', help="Choose the model type")

    parser.add_argument("--threshold", type=float, default=0.5, help="Mask threshold")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.opt == 'vis':
        visualize(args)
    else:
        test(args)