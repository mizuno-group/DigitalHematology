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
            wbc_qc=True,
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
        self.wbc_qc = wbc_qc
    
        # load WSI
        self.OS = openslide.OpenSlide(self.slide_path)
        dims = self.OS.dimensions  # (63693, 88784)

        self.coords_candi = []
        self.bg_ratio_res = []
        self.total_wbc_num = 0
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
                        if self.wbc_qc:# I'll put it in here for once.
                            wbc_num = pickup_wbc_thresh(img, size=10, blue_thresh=175)
                            if len(wbc_num) == 0:
                                pass
                            else:
                                self.coords_candi.append((x,y))
                                self.total_wbc_num = self.total_wbc_num + len(wbc_num)
                        else:
                            self.coords_candi.append((x,y))
                    else:
                        pass
                else:
                    self.coords_candi.append((x,y))

        print("Total Patches: {}".format(len(self.bg_ratio_res)))
        print("Final Patches (passed QC): {}".format(len(self.coords_candi)))
        if self.wbc_qc:
            print("Total number of WBC detected: {}".format(self.total_wbc_num))
    
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

def pickup_wbc_thresh(wsi_test, size=60, blue_thresh=150, red_thresh=150):
    image_array = np.array(wsi_test) #Conversion to np.array
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) #gray scale

    ret, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV) #OTSU

    img_pro = np.array([((np.array(image_array[i]).T)*(np.array(binary[i])*np.array([1/255]))).T for i in range(len(image_array))]) #OTSU+color Image
    pil_img = Image.fromarray(img_pro.astype(np.uint8))

    x = 0
    y = 0
    X = [0, 0]
    Y = [0, 0]
    blue_mean_max = 0

    loc_lst = []
    ap = loc_lst.append

    for i in range(0, image_array.shape[0], 10):
        for j in range(0, image_array.shape[1], 10):
            pil_img_batch = pil_img.crop((i, j, i+size, j+size))
            numpy_image = np.array(pil_img_batch)
            blue_mean = numpy_image[:, :, 2].mean()
            red_mean = numpy_image[:, :, 0].mean()
            if blue_mean > blue_thresh and red_mean < red_thresh:
                ap([i, j])
            else:
                pass
    area_lst = []
    ap_area = area_lst.append
    del_area = area_lst.remove

    for loc in loc_lst:
        for area in area_lst:
            if (area[0][0]-2*size <= loc[0] <= area[0][1]+2*size) and (area[1][0]-2*size <= loc[1] <= area[1][1]+2*size):
                xmin = min(loc[0], area[0][0])
                xmax = max(loc[0]+size, area[0][1])
                ymin = min(loc[1], area[1][0])
                ymax = max(loc[1]+size, area[1][1])
                del_area(area)
                ap_area([[xmin, xmax], [ymin, ymax]])
                break
        else:
            ap_area([[loc[0], loc[0]+size], [loc[1], loc[1]+size]])
    pickup_wbc_lst = [i for i in area_lst if i[0][1]-i[0][0] > 3*size and i[1][1]-i[1][0] > 3*size]
    return pickup_wbc_lst
