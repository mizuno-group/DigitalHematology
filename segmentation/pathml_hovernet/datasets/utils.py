"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import itertools
import numpy as np


def pannuke_multiclass_mask_to_nucleus_mask(multiclass_mask):
    """
    Convert multiclass mask from PanNuke to a single channel nucleus mask.
    Assumes each pixel is assigned to one and only one class. Sums across channels, except the last mask channel
    which indicates background pixels in PanNuke.
    Operates on a single mask.

    Args:
        multiclass_mask (torch.Tensor): Mask from PanNuke, in classification setting. (i.e. ``nucleus_type_labels=True``).
            Tensor of shape (6, 256, 256).

    Returns:
        Tensor of shape (256, 256).
    """
    # verify shape of input
    assert (
        multiclass_mask.ndim == 3 and multiclass_mask.shape[0] == 6
    ), f"Expecting a mask with dims (6, 256, 256). Got input of shape {multiclass_mask.shape}"
    assert (
        multiclass_mask.shape[1] == 256 and multiclass_mask.shape[2] == 256
    ), f"Expecting a mask with dims (6, 256, 256). Got input of shape {multiclass_mask.shape}"
    # ignore last channel
    out = np.sum(multiclass_mask[:-1, :, :], axis=0)
    return out

def stack_mask(mask, mask_dic, wbc_multi=True, bg_label=0):
    stack_layers = []
    # RBC layer
    rbc_m = min(list(itertools.chain.from_iterable(mask_dic.values())))-1
    rbc_layer = np.where(mask<=rbc_m,mask,0)
    stack_layers.append(rbc_layer)  # add RBC layer

    # WBC layer
    if wbc_multi:
        for i,k in enumerate(mask_dic):
            tmp_labels = mask_dic.get(k)
            tmp_layer = np.where((mask>=min(tmp_labels)) & (mask<=max(tmp_labels)),mask,0)
            stack_layers.append(tmp_layer)  # add each WBC layer
    else:
        all_labels = list(itertools.chain.from_iterable(mask_dic.values()))
        tmp_layer = np.where((mask>=min(all_labels)) & (mask<=max(all_labels)),mask,0)
        stack_layers.append(tmp_layer)  # add each WBC layer

    # Background layer
    bg_binary = np.where(mask==bg_label,1,0)  # set -1
    stack_layers.append(bg_binary)

    # Stack and 
    final_mask = np.stack(stack_layers)  # (C, H, W)

    return final_mask