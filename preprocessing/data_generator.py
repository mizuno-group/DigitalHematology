# -*- coding: utf-8 -*-
"""
Created on 2024-04-05 (Fri) 17:37:40

@author: I.Azuma
"""
# %%
import cv2
import copy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image

# %%
def fixCell2Bg(sample=None, mask=None, bg=None, tlX=None, tlY=None):
    if sample is None or mask is None or bg is None or tlX is None or tlY is None:
        raise ValueError( 'ERROR: one or more input arguments missing ' \
               'in fixSampleToBg. Aborting.' )
    bgH, bgW, _ = bg.shape
    sample = np.array(sample,dtype=np.uint8)
    sampleH, sampleW, _ = sample.shape # Size of the 

    brY, brX = min(tlY + sampleH, bgH), min(tlX + sampleW, bgW) # the bottom right corner
    bgRegionToBeReplaced = bg[tlY : brY, tlX : brX, :]
    bgRegTBRh, bgRegTBRw, _ = bgRegionToBeReplaced.shape # size after boundary consideration
    mask_on_region = mask[0:bgRegTBRh, 0:bgRegTBRw]

    bgRegionToBeReplaced[mask_on_region==1]=[0,0,0]
    boundingRegion = np.asarray(bgRegionToBeReplaced, dtype=np.uint8 )
    boundingRegion = boundingRegion[0:bgRegTBRh, 0:bgRegTBRw, :]

    onlyObjectRegionOfSample = copy.deepcopy(sample)
    onlyObjectRegionOfSample[mask==0]=[0,0,0]
    onlyObjectRegionOfSample = onlyObjectRegionOfSample[0 : bgRegTBRh, 0 : bgRegTBRw, :]

    # Fix
    img = copy.deepcopy(bg)
    img[tlY:brY, tlX:brX, :] = onlyObjectRegionOfSample + boundingRegion

    # Center pixel of the region
    posY = round( (brY + tlY) * 0.5 )
    posX = round( (brX + tlX) * 0.5 )
    bboxH = brY - tlY
    bboxW = brX - tlX

    return img, posX, posY, bboxW, bboxH