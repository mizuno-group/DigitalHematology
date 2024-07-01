# -*- coding: utf-8 -*-
"""
Created on 2024-04-18 (Thu) 23:23:22

Pseudo smear image generator for instance segmentation.

@author: I.Azuma
"""
import cv2
import copy
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from glob import glob
from PIL import Image
from tqdm import tqdm

# %% Basic functions
def fixCell2Bg(sample=None, mask=None, bg=None, tlX=None, tlY=None, label_matrix=None,label_v=1):
    if sample is None or mask is None or bg is None or tlX is None or tlY is None:
        raise ValueError( 'ERROR: one or more input arguments missing ' \
               'in fixSampleToBg. Aborting.' )
    bgH, bgW, _ = bg.shape
    sample = np.array(sample,dtype=np.uint8)
    sampleH, sampleW, _ = sample.shape # Size of the 

    brY, brX = min(tlY + sampleH, bgH), min(tlX + sampleW, bgW) # the bottom right corner
    bgRegionToBeReplaced = bg[tlY : brY, tlX : brX, :]
    bgRegTBRh, bgRegTBRw, _ = bgRegionToBeReplaced.shape # size after boundary consideration

    mask = np.array(mask,dtype=np.uint8)
    if len(mask.shape) == 3:
        mask = np.where(mask.mean(axis=-1)>0,1,0) # Convert to binary

    mask_on_region = mask[0:bgRegTBRh, 0:bgRegTBRw]

    bgRegionToBeReplaced[mask_on_region==1]=[0,0,0]
    boundingRegion = np.asarray(bgRegionToBeReplaced, dtype=np.uint8)
    boundingRegion = boundingRegion[0:bgRegTBRh, 0:bgRegTBRw, :]

    onlyObjectRegionOfSample = copy.deepcopy(sample)
    onlyObjectRegionOfSample[mask==0]=[0,0,0]
    onlyObjectRegionOfSample = onlyObjectRegionOfSample[0 : bgRegTBRh, 0 : bgRegTBRw, :]

    # Affix on image
    img = copy.deepcopy(bg)
    img[tlY:brY, tlX:brX, :] = onlyObjectRegionOfSample + boundingRegion

    # Affix on mask
    if label_matrix is None:  # initial trial
        label_matrix = np.zeros((img.shape[0],img.shape[1]), dtype=int)

    objectRegionLabel = np.where(bgRegionToBeReplaced.sum(axis=-1)==0,label_v,0)
    label_mat = copy.deepcopy(label_matrix)
    label_mat = label_mat[tlY:brY, tlX:brX]
    label_mat[objectRegionLabel==label_v]=0

    label_matrix[tlY:brY, tlX:brX] = label_mat + objectRegionLabel

    # Center pixel of the region
    posY = round( (brY + tlY) * 0.5 )
    posX = round( (brX + tlX) * 0.5 )
    bboxH = brY - tlY
    bboxW = brX - tlX

    return img, label_matrix, posX, posY, bboxW, bboxH


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

# Add margin
def add_margin(pil_img, margin, color):
    top = margin[0]
    right = margin[1]
    bottom = margin[2]
    left = margin[3]
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def lisc_processor(cell_pilimg,mask_pilimg,margin_size=(4,4,4,4),do_imshow=False):
    """ Data processing for LISC dataset

    There is a size difference between the image (576x720) and annotation (568x712).
    Extend the annotation (mask) image with marginal space.

    Args:
        cell_pilimg (PIL.Image): Image file loaded with PIL.
        mask_pilimg (PIL.Image): Annotation file loaded with PIL.
        margin_size (tuple, optional): Margin for matching the size. Defaults to (4,4,4,4).
        do_imshow (bool, optional): Defaults to False.

    """
    # Adjust the annotation image size
    mask_image = add_margin(mask_pilimg, margin_size, (0, 0, 0))  # add margin

    mask_array = np.array(mask_image,dtype=np.uint8)
    cell_array = np.array(cell_pilimg,dtype=np.uint8)

    # Confirm the size match
    if cell_array.shape != mask_array.shape:
        print("sample: {} \nmask: {}".format(sample.shape, mask_array.shape))
        raise ValueError("Size Mismatch")
    
    # Extract patch that includes masked region
    cell_posi = np.where(mask_array.sum(axis=-1)>0)
    upper = cell_posi[0].min()
    lower = cell_posi[0].max()
    left = cell_posi[1].min()
    right = cell_posi[1].max()

    #crop_cell = cell_array[upper:lower,left:right,:]
    #crop_mask = mask_array[upper:lower,left:right,:]
    crop_cell = cell_pilimg.crop((left,upper,right,lower))
    crop_mask = mask_pilimg.crop((left,upper,right,lower))

    # Visualization
    if do_imshow:
        plt.imshow(crop_cell)
        plt.show()
        plt.imshow(crop_mask)
        plt.show()

    return crop_cell, crop_mask

# %% Pipeline
def gen_bg_single(rbc_candi:list,random_sets:list,bgH=1000,bgW=1000,sampleH=30,sampleW=30,n_iter=10000,n_cells=500,tol=40,bg=None):
    if bg is None:
        bg = np.ones((bgH,bgW,3))*255  # blank image
    bg = np.array(bg,dtype=np.uint8)

    cell_count = 0
    pos_history = [(0,0)]
    for nc in range(n_iter):
        # Load raw image
        random.seed(random_sets[nc])
        rbc_file = random.sample(rbc_candi,1)[0]
        rbc_image = Image.open(rbc_file).convert("RGB")
        rbc_image = rbc_image.resize((sampleW,sampleH))  # resize
        sample = np.array(rbc_image,dtype=np.uint8)

        # Rotation
        rotation_list = [0,45,90,135,180,225,270,315]
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
            if cell_count == 0:
                bg, label_matrix, posX, posY, bboxW, bboxH = fixCell2Bg(sample=sample, mask=mask, bg=bg, tlX=bgTlX, tlY=bgTlY, label_matrix=None, label_v=cell_count+1)
            else:
                bg, label_matrix, posX, posY, bboxW, bboxH = fixCell2Bg(sample=sample, mask=mask, bg=bg, tlX=bgTlX, tlY=bgTlY, label_matrix=label_matrix, label_v=cell_count+1)

            pos_history.append((posX,posY))
            cell_count += 1

            if cell_count == n_cells:
                print("{} cells have been collected.".format(n_cells))
                break
            
    return bg, label_matrix

def affix_lisc(bg,random_sets:list,img_file_path='/Path/To/Main Dataset',
               mask_file_path='/Path/To/Ground Truth Segmentation',
               pos_history=None,label_matrix=None,
               cell_type='eosi',sampleH=50,sampleW=50,n_iter=1000,n_cells=10,tol=40):
    type_candi = ['Baso','eosi','lymp','mono','neut']

    # These cell IDs contain multiple cells
    multicell_ids = [['7','31'],
                     [],
                     ['3','20','23','33','34','45','52'],
                     [],
                     ['31','38','42','47']
                     ]
    multicells = multicell_ids[type_candi.index(cell_type)]

    if cell_type not in type_candi:
        raise ValueError('Inappropriate cell type. Choose from {}'.format(type_candi))
    
    if label_matrix is None:
        raise ValueError('Set label_matrix. You can obtain from the results of gen_bg_single()')
    
    # Load file path that contains image file
    wbc_candi = sorted(list(glob(img_file_path+'/{}/*.bmp'.format(cell_type))))

    cell_count = 0
    used_ids = []
    instance_label_list = []
    # Initialize the affixed cell location information
    if pos_history is None:
        pos_history = [(0,0)]
    max_label = label_matrix.max()
    for nc in range(n_iter):
        random.seed(random_sets[nc])
        img_idx = random.randint(0,len(wbc_candi)-1)
        wbc_file = wbc_candi[img_idx]  # Collect file path
        cell_id = wbc_file.split('/')[-1].split('.')[0] # Extract the cell ID
        
        # If multiple cells are included, they are not appropriate 
        # for the present analysis and will be skipped.
        if cell_id in multicells:
            continue

        wbc_mask_candi = sorted(list(glob(mask_file_path+'/{}/areaforexpert1/{}_expert*.bmp'.format(cell_type, str(cell_id)))))  # Seek the mask image corresponding to the loaded cell image.
        if len(wbc_mask_candi)!=1:
            print(wbc_file)
            print(wbc_mask_candi)
            raise ValueError("Something is wrong.")
        # Load images
        cell_image = Image.open(wbc_file).convert("RGB")
        mask_image = Image.open(wbc_mask_candi[0]).convert("RGB")

        # Preprocessing
        crop_cell, crop_mask = lisc_processor(cell_image,mask_image,margin_size=(4,4,4,4),do_imshow=False)

        # Resize
        crop_cell = crop_cell.resize((sampleW,sampleH))
        crop_mask = crop_mask.resize((sampleW,sampleH))

        # Convert to array
        sample = np.array(crop_cell,dtype=np.uint8)
        mask = np.array(crop_mask,dtype=np.uint8)

        # Rotation
        rotation_list = [0,90,180,270]
        angle = random.sample(rotation_list,1)[0]
        M = cv2.getRotationMatrix2D((sampleW//2, sampleH//2), angle, 1)
        sample = cv2.warpAffine(sample, M, (sampleW, sampleH))
        sample = np.array(sample,dtype=np.uint8)

        # Mask preparation
        mask = cv2.warpAffine(mask, M, (sampleW, sampleH))
        mask = np.array(mask,dtype=np.uint8)
    
        # Set location
        bgW, bgH, _ = bg.shape
        bgTlX = random.randint(0,bgW)
        bgTlY = random.randint(0,bgH) 

        # Calculate distance from the nearest point
        tmp = [np.linalg.norm(np.array((bgTlX,bgTlY)) - np.array(t)) for t in pos_history]
        minidx = np.argmin(tmp)

        if min(tmp) < tol:
            # Too close to already located cells.
            pass
        else:
            label_v = max_label+cell_count+1
            bg, label_mat, posX, posY, bboxW, bboxH = fixCell2Bg(sample=sample, mask=mask, bg=bg, tlX=bgTlX, tlY=bgTlY, label_matrix=label_matrix, label_v=label_v)
            pos_history.append((posX,posY))
            cell_count += 1
            used_ids.append(cell_id)
            instance_label_list.append(label_v)

            # When you achieve the goal
            if cell_count == n_cells:
                #print("{} cells have been collected.".format(n_cells))
                break
    
    #print('Used Cells: {}'.format(used_ids))

    return bg, label_mat, pos_history, instance_label_list
