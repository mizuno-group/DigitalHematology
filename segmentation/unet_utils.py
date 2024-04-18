# -*- coding: utf-8 -*-
"""
Created on 2024-04-12 (Fri) 10:19:42

UNet model

Reference
- https://zenn.dev/aidemy/articles/a43ebe82dfbb8b
- https://www.kaggle.com/code/shnakazawa/semantic-segmentation-with-pytorch-and-u-net
- https://qiita.com/gensal/items/03e9a6d0f7081e77ba37

@author: I.Azuma
"""
# %%
import gc
import cv2
import time
import random
import numpy as np
import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import Image
from glob import glob
from tqdm import tqdm

import torch
print(torch.cuda.get_device_name())
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import segmentation_models_pytorch as smp

import warnings
warnings.simplefilter('ignore')

print(f'PyTorch version {torch.__version__}')
print(f'Albumentations version {A.__version__}')

# %%
def transform_common():
    transforms = [
        A.Resize(256,256,p=1),
        ToTensorV2(p=1)
    ]
    return A.Compose(transforms)

class MySmearDataset(Dataset):
    def __init__(self, data_root='/workspace/HDDX/Azuma/Hematology/results/240408_PseudoSmearImage/240409_pseudo_image/240409_C7thinF_Processed', transforms=None, stage='train', compression=0.3, return_mask=True,gray=True):
        super().__init__()
        self.data_root = data_root
        self.transforms = transforms
        self.stage = stage
        self.return_mask = return_mask
        self.gray = gray

        whole_image_path_list = sorted(list(glob(f'{self.data_root}/Image/{stage}/*.npy')))
        
        self.image_path_list = whole_image_path_list[0:int(len(whole_image_path_list)*compression)]

        if self.return_mask:
            whole_mask_path_list = sorted(list(glob(f'{self.data_root}/Label/{stage}/*.npy')))
            self.mask_path_list = whole_mask_path_list[0:int(len(whole_mask_path_list)*compression)]

            if len(self.image_path_list) != len(self.mask_path_list):
                raise ValueError("The number of image and mask do not match.")

    def __len__(self):
        return len(self.image_path_list)
    
    def __getitem__(self, index):
        # Extract IDs
        image_id = self.image_path_list[index].split('_')[-1].split('.npy')[0]  # e.g. Img_C7thinF_890.npy >> 890

        # Load images
        image = np.load(self.image_path_list[index]).astype(np.float32)
        if self.gray:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image /= 255.0 # normalization


        if self.return_mask:
            # Load masks
            mask_mat = np.load(self.mask_path_list[index])
            #mask_mat = np.where(mask_mat>0,1,0)  # binary mask

            # Transform images and masks
            if self.transforms:
                transformed = self.transforms(image=image, mask=mask_mat)
                image, mask_mat = transformed['image'], transformed['mask']
            
            # Convert one hot vector to evaluate segmentation performance
            mask_mat = F.one_hot(torch.tensor(mask_mat).long(), num_classes=mask_mat.max()+1)
            mask_mat = np.array(mask_mat.permute(2,0,1))
            
            return image, mask_mat, image_id

        else:
            # Transform images and masks
            if self.transforms:
                transformed = self.transforms(image=image)
                image = transformed['image']

            return image, image_id

class MySmearInfDataset(Dataset):
    def __init__(self, data_root='/workspace/HDDX/Azuma/Hematology/results/240408_PseudoSmearImage/240409_pseudo_image/240409_C7thinF_Processed', transforms=None, gray=True):
        super().__init__()
        self.data_root = data_root
        self.transforms = transforms
        self.gray = gray

        whole_image_path_list = sorted(list(glob(f'{self.data_root}/*.npy')))
        
        self.image_path_list = whole_image_path_list

    def __len__(self):
        return len(self.image_path_list)
    
    def __getitem__(self, index):
        # Extract IDs
        image_id = self.image_path_list[index].split('_')[-1].split('.npy')[0]  # e.g. Img_C7thinF_890.npy >> 890

        # Load images
        image = np.load(self.image_path_list[index]).astype(np.float32)
        if self.gray:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image /= 255.0 # normalization

        # Transform images and masks
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']

        return image, image_id


# %%
class DoubleConv(nn.Module):
    """DoubleConv is a basic building block of the encoder and decoder components. 
    Consists of two convolutional layers followed by a ReLU activation function.
    """    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class Down(nn.Module):
    """Downscaling.
    Consists of two consecutive DoubleConv blocks followed by a max pooling operation.
    """    
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x


class DoubleConv(nn.Module):
    """DoubleConv is a basic building block of the encoder and decoder components. 
    Consists of two convolutional layers followed by a ReLU activation function.
    """    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class Down(nn.Module):
    """Downscaling.
    Consists of two consecutive DoubleConv blocks followed by a max pooling operation.
    """    
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x


class Up(nn.Module):
    """Upscaling.
    Performed using transposed convolution and concatenation of feature maps from the corresponding "Down" operation.
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input tensor shape: (batch_size, channels, height, width)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
    

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        
        self.down4 = Down(512,1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        #x = torch.sigmoid(x)
        
        return x
