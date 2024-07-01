# -*- coding: utf-8 -*-
"""
2024/7/1

@author: T.Iwasaka
"""

from PIL import Image
import matplotlib.pyplot as plt 
import cv2
import numpy as np

from openslide import OpenSlide
from matplotlib.collections import LineCollection

from pyclustering.cluster.gmeans import gmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

import warnings
np.warnings = warnings

def pickup_wbc_thresh(wsi_test, size=60, blue_thresh=150, red_thresh=150):
    image_array = np.array(wsi_test) #Conversion to np.array
    print(image_array.shape)
    pil_img = Image.fromarray(image_array) #PIL Image

    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) #gray scale

    ret, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV) #OTSU
    print("OHTSU Thresh :", ret)

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
    pickup_wbc_lst = [[[i[0][0]-size/2, i[1][0]-size/2], 
    [i[0][0]-size/2, i[1][1]+size/2], 
    [i[0][1]+size/2, i[1][1]+size/2], 
    [i[0][1]+size/2, i[1][0]-size/2], 
    [i[0][0]-size/2, i[1][0]-size/2]] for i in area_lst if i[0][1]-i[0][0] > 3*size and i[1][1]-i[1][0] > 3*size]
    wbc_area = np.array(pickup_wbc_lst)
    lc = LineCollection(wbc_area, color='lime', linestyle='solid', linewidths=3)
    fig, ax = plt.subplots()
    ax.imshow(image_array)
    ax.add_collection(lc)
    plt.title("WBC detection")
    plt.show()

def pickup_wbc_rbc_thresh(wsi_test, wbc_patch=10, rbc_patch=4, blue_thresh=175, red_thresh=150):
    # Background separation of images
    image_array = np.array(wsi_test) # Conversion to np.array
    print(image_array.shape)
    pil_img = Image.fromarray(image_array) #PIL Image
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) #gray scale
    ret, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV) #OTSU
    print("OHTSU Thresh :", ret)
    img_pro = np.array([((np.array(image_array[i]).T)*(np.array(binary[i])*np.array([1/255]))).T for i in range(len(image_array))]) #OTSU+color Image
    pil_img = Image.fromarray(img_pro.astype(np.uint8))

    # Extraction of patches containing leukocytes
    wbc_loc_lst = []
    ap_wbc = wbc_loc_lst.append
    for i in range(0, image_array.shape[0], wbc_patch):
        for j in range(0, image_array.shape[1], wbc_patch):
            pil_img_batch = pil_img.crop((i, j, i+wbc_patch, j+wbc_patch))
            numpy_image = np.array(pil_img_batch)
            blue_mean = numpy_image[:, :, 2].mean()
            red_mean = numpy_image[:, :, 0].mean()
            if blue_mean > blue_thresh and red_mean < red_thresh:
                ap_wbc([i, j])
            else:
                pass
    # Determination of leukocyte regions
    area_lst = []
    ap_area = area_lst.append
    del_area = area_lst.remove
    for loc in wbc_loc_lst:
        for area in area_lst:
            if (area[0][0]-2*wbc_patch <= loc[0] <= area[0][1]+2*wbc_patch) and (area[1][0]-2*wbc_patch <= loc[1] <= area[1][1]+2*wbc_patch):
                xmin = min(loc[0], area[0][0])
                xmax = max(loc[0]+wbc_patch, area[0][1])
                ymin = min(loc[1], area[1][0])
                ymax = max(loc[1]+wbc_patch, area[1][1])
                del_area(area)
                ap_area([[xmin, xmax], [ymin, ymax]])
                break
        else:
            ap_area([[loc[0], loc[0]+wbc_patch], [loc[1], loc[1]+wbc_patch]])
    pickup_wbc_lst = [[[i[0][0]-wbc_patch/2, i[1][0]-wbc_patch/2], 
    [i[0][0]-wbc_patch/2, i[1][1]+wbc_patch/2], 
    [i[0][1]+wbc_patch/2, i[1][1]+wbc_patch/2], 
    [i[0][1]+wbc_patch/2, i[1][0]-wbc_patch/2], 
    [i[0][0]-wbc_patch/2, i[1][0]-wbc_patch/2]] for i in area_lst if i[0][1]-i[0][0] > 3*wbc_patch and i[1][1]-i[1][0] > 3*wbc_patch]
    wbc_area = np.array(pickup_wbc_lst)
    lc = LineCollection(wbc_area, color='deepskyblue', linestyle='solid', linewidths=3)

    # Extraction of patches containing erythrocytes
    loc_clst = []
    ap_rbc = loc_clst.append
    for i in range(0, image_array.shape[0], rbc_patch):
        for j in range(0, image_array.shape[1], rbc_patch):
            pil_img_batch = pil_img.crop((i, j, i+rbc_patch, j+rbc_patch))
            numpy_image = np.array(pil_img_batch)
            blue_mean = numpy_image[:, :, 2].mean()
            red_mean = numpy_image[:, :, 0].mean()
            if blue_mean < blue_thresh and red_mean > red_thresh:
                ap_rbc([i, j])
            elif blue_mean > 0 and red_mean > red_thresh:
                ap_rbc([i, j])
            else:
                pass
    # Execution of G-means
    X_2 = np.array([[i[0], i[1]] for i in loc_clst])
    gm_c = kmeans_plusplus_initializer(X_2, 2).initialize()
    gm_i = gmeans(data=X_2, initial_centers=gm_c, kmax=10, ccore=False, random_state=24771)
    gm_i.process()
    z_gm = np.ones(X_2.shape[0])
    for k in range(len(gm_i._gmeans__clusters)):
        z_gm[gm_i._gmeans__clusters[k]] = k+1
    centers = np.array(gm_i._gmeans__centers)

    # Erythrocyte determination (removal of duplicates, exclusion of items within leukocyte areas)
    new_loc = []
    ap_newloc = new_loc.append
    del_loc = new_loc.remove
    for i in centers:
        for j in new_loc:
            if np.linalg.norm(i - j) < 20: # Likely successful values
                x = (i[0]+j[0])/2
                y = (i[1]+j[1])/2
                ap_newloc([x, y])
                del_loc([j[0], j[1]])
                break
        else:
            ap_newloc([i[0], i[1]])
    wbc_area_loc = [[[i[0][0]+wbc_patch/2, i[2][0]-wbc_patch/2], [i[0][1]+wbc_patch/2, i[2][1]-wbc_patch/2]]  for i in wbc_area]
    for loc in new_loc:
        for area in wbc_area_loc:
            if area[0][0] < loc[0] < area[0][1] and area[1][0] < loc[1] < area[1][1]:
                del_loc(loc)
    new_loc = np.array(new_loc)

    # Visualization
    fig, ax = plt.subplots()
    plt.scatter(new_loc[:,0],new_loc[:,1],s=100, marker='h',c='orangered')
    ax.imshow(image_array)
    ax.add_collection(lc)
    plt.title("WBC & RBC detection")
    plt.show()