# -*- coding: utf-8 -*-
"""
Created on 2024-02-06 (Tue) 17:25:11

WSI handler

References
- https://github.com/vqdang/hover_net/blob/master/misc/wsi_handler.py

@author: I.Azuma
"""
# %%
import numpy as np

import openslide
from openslide import OpenSlide
from openslide import OpenSlideError

from PIL import Image

# %%
def image_generator_slide(slide_path,height=512,width=512):
    OS = openslide.OpenSlide(slide_path)
    dim = OS.dimensions
    for x in range(0,dim[0],height):
        for y in range(0,dim[1],width):
            try:
                im = OS.read_region((x,y),0,(height,width))
                im = np.array(im)
                im = im[:,:,:3]
                yield im,'{},{}'.format(x,y)
            except OpenSlideError as error:
                OS = openslide.OpenSlide(slide_path)

def generator(slide_path, height, width):
    G = image_generator_slide(
        slide_path,height,width)
    for image,coords in G:
        #image = image / 255
        yield image,coords

