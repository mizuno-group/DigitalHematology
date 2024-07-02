# -*- coding: utf-8 -*-
"""
Created on 2024-07-02 (Tue) 15:44:11



@author: I.Azuma
"""
# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import openslide
from openslide import OpenSlide, OpenSlideError

# %%
class SmearDatasetQC(Dataset):
    def __init__(
            self,
            slide_path:str,
            h = 1024,
            w = 1024,
            extra_padding = 128,
            x_start = 30000,
            x_end = 32000,
            y_start = 78000,
            y_end = 79000,
            qc=True,
        ):

        self.slide_path = slide_path
        self.h = h
        self.w = w
        self.extra_padding = extra_padding
        self.x_start = x_start
        self.y_start = y_start
        self.x_end = x_end
        self.y_end = y_end
        self.qc = qc
    
        # load WSI
        self.OS = openslide.OpenSlide(self.slide_path)
        dims = self.OS.dimensions  # (63693, 88784)

        self.coords_candi = []
        self.bg_ratio_res = []
        for x1 in range(x_start,x_end,h):
            for y1 in range(y_start,y_end,w):
                
                x = x1 - extra_padding
                y = y1 - extra_padding

                x = np.maximum(0,x)  # avoid negative value
                y = np.maximum(0,y)  # avoid negative value

                if x + h + (2*extra_padding) > dims[0]:
                    x = dims[0] - h - (extra_padding*2)  # avoid off-screen
                if y + w + (2*extra_padding) > dims[1]:
                    y = dims[1] - w - (extra_padding*2)  # avoid off-screen

                # Quality Check (QC) with Otsu method
                """ Quality Check 
                1. Background ratio.
                2. Inclusion of white blood cells. << Not yet.
                """
                if self.qc:
                    img = self.OS.read_region(
                    (x,y),0,
                    (self.h+(self.extra_padding*2),self.w+(self.extra_padding*2)))
                    bg_ratio = calc_bgratio(img=img,bg_label=0)
                    self.bg_ratio_res.append(bg_ratio)
                    
                    if 0.45 < bg_ratio < 0.55:  # NOTE: this is hard threshold
                        self.coords_candi.append((x,y))
                    else:
                        pass
                
                else:
                    self.coords_candi.append((x,y))

        print("Total Patches: {}".format(len(self.bg_ratio_res)))
        print("Final Patches (passed QC): {}".format(len(self.coords_candi)))
    
    def __len__(self):
        return len(self.coords_candi)
    
    def __getitem__(self, idx):
        coords = self.coords_candi[idx]  # (x,y)
        image = self.OS.read_region(
            coords,0,
            (self.h+(self.extra_padding*2),self.w+(self.extra_padding*2)))
        image = np.array(image)[:,:,:3]  # (h+ext*2,w+ext*2,3)

        return image, coords

# %% Functions
def calc_bgratio(img_path:str='',img=None,bg_label=0):
    """ Perform Otsu method and calc background ratio.

    Parameters
    ----------
    img_path : str
        Path to the target image.
    bg_label : int, optional
        Background label, by default 0
        
    """
    if img is None:
        img = Image.open(img_path).convert("RGB") 
    img = img.convert("RGB")
    gray = img.convert("L")  # convert to gray scale

    # Otsu method
    ret, bin_img = cv2.threshold(np.array(gray), 10, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # remove object area
    bg_label = 0
    bg_ratio = (bin_img==bg_label).sum() / (bin_img.shape[0]*bin_img.shape[1])

    return bg_ratio
