# -*- coding: utf-8 -*-
"""
Created on 2024-04-05 (Fri) 17:37:40

@author: I.Azuma
"""
# %%
import cv2
import copy
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image

# %% Basic functions
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


def datasetMeanStd(dataDir=None):
    '''
    Takes in the location of the images as input.
    Calculates the mean and std of the images of a dataset that is needed 
    to normalize the images before training.
    Returns the mean and std in the form of float arrays 
    (e.g. mean = [ 0.52, 0.45, 0.583 ], std = [ 0.026, 0.03, 0.0434 ] )
    '''
    sampleH, sampleW = 30, 30
    meanOfImg = np.zeros((sampleW, sampleH, 3), dtype=np.float32)
    meanOfImgSquare = np.zeros((sampleW, sampleH, 3), dtype=np.float32)
    listOfImg = glob(dataDir+'/*.png')
    nImg = len(listOfImg)
    
    for idx, path in enumerate(listOfImg):
        img = Image.open(path).convert("RGB")
        img = img.resize((sampleW,sampleH)) # Resize
        img = np.array(img,dtype=np.uint8)
        
        meanOfImg += img / nImg
        meanOfImgSquare += img * ( img / nImg )
    
    # Now taking mean of all pixels in the mean image created in the loop.
    # Now meanOfImg is 224 x 224 x 3.
    meanOfImg = np.mean( meanOfImg, axis=0 )
    meanOfImgSquare = np.mean( meanOfImgSquare, axis=0 )
    # Now meanOfImg is 224 x 3.
    meanOfImg = np.mean( meanOfImg, axis=0 )
    meanOfImgSquare = np.mean( meanOfImgSquare, axis=0 )
    # Now meanOfImg is 3.
    variance = meanOfImgSquare - meanOfImg * meanOfImg
    std = np.sqrt( variance )
    
    return meanOfImg, std

# %% Pipeline
def gen_single(rbc_candi:list,random_sets:list,bgH=1000,bgW=1000,sampleH=30,sampleW=30,n_iter=10000,n_cells=500,tol=40):
    bg = np.ones((bgH,bgW,3))*255 # Blank image
    bg = np.array(bg,dtype=np.uint8)

    cell_count = 0
    pos_history = [(0,0)]
    for nc in range(n_iter):
        # Load raw image
        random.seed(random_sets[nc])
        rbc_file = random.sample(rbc_candi,1)[0]
        rbc_image = Image.open(rbc_file).convert("RGB")
        rbc_image = rbc_image.resize((sampleW,sampleH)) # Resize
        sample = np.array(rbc_image,dtype=np.uint8)

        # Rotation
        rotation_list = [0,45,90,180,225,270,315]
        angle = random.sample(rotation_list,1)[0]
        M = cv2.getRotationMatrix2D((sampleW//2, sampleH//2), angle, 1)
        sample = cv2.warpAffine(sample, M, (sampleW, sampleH))
        sample = np.array(sample,dtype=np.uint8)

        # Maks preparation
        sum_sample = sample.sum(axis=-1)
        mask = np.where(sum_sample>0,1,0)
        mask = np.asarray(mask, dtype=np.uint8)

        # Set location
        bgTlX = random.randint(0,bgW)
        bgTlY = random.randint(0,bgH) 

        # Calculate distance from the nearest point
        tmp = [np.linalg.norm(np.array((bgTlX,bgTlY)) - np.array(t)) for t in pos_history]
        minidx = np.argmin(tmp)

        if min(tmp) < tol:
            pass
        else:
            bg, posX, posY, bboxW, bboxH = fixCell2Bg(sample=sample, mask=mask, bg=bg, tlX=bgTlX, tlY=bgTlY)
            pos_history.append((posX,posY))
            cell_count += 1

            if cell_count == n_cells:
                print("{} cells have been collected.".format(n_cells))
                break
    return bg

