# -*- coding: utf-8 -*-
"""
Created on 2024-04-19 (Fri) 13:36:18

@author: I.Azuma
"""
# %%
import numpy as np
import pandas as pd

from glob import glob
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

import torch

from pathml_hovernet.datasets.utils import stack_mask
from pathml_hovernet.ml.hovernet import compute_hv_map

# %%
class MySmearDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transforms=None,
        stage=None,
        compression=1.0,
        bg_label=0,
        wbc_multi=True
    ):
        self.data_dir = data_dir
        self.transforms = transforms
        self.stage = stage
        self.bg_label = bg_label
        self.wbc_multi = wbc_multi

        data_dir = Path(data_dir)

        # dirs for images, masks
        if self.stage is None:
            imdir = data_dir / "Images"
            maskdir = data_dir / "Labels"
            maskdictdir = data_dir / "Label_dict"
        else:
            imdir = data_dir / "Images/{}".format(stage)
            maskdir = data_dir / "Labels/{}".format(stage)
            maskdictdir = data_dir / "Label_dict/{}".format(stage)
        
        # stop if the images and masks directories don't already exist
        assert imdir.is_dir(), f"Error: 'images' directory not found: {imdir}"
        assert maskdir.is_dir(), f"Error: 'masks' directory not found: {maskdir}"

        paths = list(imdir.glob("*"))

        self.imdir = imdir
        self.maskdir = maskdir
        self.maskdictdir = maskdictdir
        paths = [p.stem for p in paths]
        self.paths = paths[0:int(len(paths)*compression)]
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, ix):
        stem = self.paths[ix].split('_')[-1]
        impath = list(self.imdir.glob(f"*_{stem}.npy"))[0]
        maskpath = list(self.maskdir.glob(f"*_{stem}.npy"))[0] #self.maskdir / f"_{stem}.npy"
        maskdictpath = list(self.maskdictdir.glob(f"*_{stem}.pkl"))[0]  #self.maskdictdir / f"_{stem}.pkl"

        im = np.load(str(impath))
        mask_1c = np.load(str(maskpath))  # (H, W)
        mask_dic = pd.read_pickle(str(maskdictpath))

        mask = stack_mask(mask=mask_1c, mask_dic=mask_dic, wbc_multi=self.wbc_multi, bg_label=self.bg_label)  # (C, H, W)

        if self.transforms is not None:
            transformed = self.transforms(image=im, mask=mask)
            im = transformed["image"]
            mask = transformed["mask"]

        # swap channel dim to pytorch standard (C, H, W)
        im = im.transpose((2, 0, 1))

        # compute hv map
        mask_1c = np.sum(mask[:-1, :, :], axis=0)  # Skip background layer
        hv_map = compute_hv_map(mask_1c)

        out = (
            torch.from_numpy(im),
            torch.from_numpy(mask),
            torch.from_numpy(hv_map),
        )

        return out


class MySmearInfDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transforms=None,
        stage=None,
        compression=1.0
    ):
        self.data_dir = data_dir
        self.transforms = transforms
        self.stage=stage

        data_dir = Path(data_dir)

        # dirs for images
        if self.stage is None:
            imdir = data_dir
        else:
            imdir = data_dir / "{}".format(stage)
        
        # stop if the images directory don't already exist
        assert imdir.is_dir(), f"Error: 'images' directory not found: {imdir}"

        paths = list(imdir.glob("*"))

        self.imdir = imdir
        paths = [p.stem for p in paths]
        self.paths = paths[0:int(len(paths)*compression)]
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, ix):
        stem = self.paths[ix].split('_')[-1]
        impath = list(self.imdir.glob(f"*_{stem}.npy"))[0]

        im = np.load(str(impath))

        if self.transforms is not None:
            transformed = self.transforms(image=im)
            im = transformed["image"]

        # swap channel dim to pytorch standard (C, H, W)
        #im = im.transpose((2, 0, 1))

        out = (
            im
        )

        return out

class MySmearCroppedDataset(Dataset):
    def __init__(
        self,
        data_dir,
        maskdict_dir=None,
        transforms=None,
        stage=None,
        compression=1.0,
        bg_label=0,
        wbc_multi=True
    ):
        self.data_dir = data_dir
        self.transforms = transforms
        self.stage=stage
        self.bg_label = bg_label
        self.wbc_multi=wbc_multi

        data_dir = Path(data_dir)
        if maskdict_dir is None:
            maskdict_dir = data_dir
        else:
            maskdict_dir = Path(maskdict_dir)

        # dirs for images, masks
        if self.stage is None:
            imdir = data_dir / "Images"
            maskdir = data_dir / "Labels"
            maskdictdir = maskdict_dir / "Label_dict"
        else:
            imdir = data_dir / "Images/{}".format(stage)
            maskdir = data_dir / "Labels/{}".format(stage)
            maskdictdir = maskdict_dir / "Label_dict/{}".format(stage)
        
        # stop if the images and masks directories don't already exist
        assert imdir.is_dir(), f"Error: 'images' directory not found: {imdir}"
        assert maskdir.is_dir(), f"Error: 'masks' directory not found: {maskdir}"

        paths = list(imdir.glob("*"))

        self.imdir = imdir
        self.maskdir = maskdir
        self.maskdictdir = maskdictdir
        paths = [p.stem for p in paths]
        self.paths = paths[0:int(len(paths)*compression)]
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, ix):
        stem = self.paths[ix].split('Cropped_')[-1]  # 0_1_25
        impath = list(self.imdir.glob(f"*_{stem}.npy"))
        assert len(impath) == 1
        maskpath = list(self.maskdir.glob(f"*_{stem}.npy"))
        assert len(maskpath) == 1
        stem2 = stem.split('_')[-1]  # 0_1_25 >> 25
        maskdictpath = list(self.maskdictdir.glob(f"*_{stem2}.pkl"))
        assert len(maskdictpath) == 1

        im = np.load(str(impath[0]))
        mask_1c = np.load(str(maskpath[0]))  # (H, W)
        mask_dic = pd.read_pickle(str(maskdictpath[0]))

        mask = stack_mask(mask=mask_1c, mask_dic=mask_dic, wbc_multi=self.wbc_multi, bg_label=self.bg_label)  # (C, H, W)

        if self.transforms is not None:
            transformed = self.transforms(image=im, mask=mask)
            im = transformed["image"]
            mask = transformed["mask"]

        # swap channel dim to pytorch standard (C, H, W)
        im = im.transpose((2, 0, 1))

        # compute hv map
        mask_1c = np.sum(mask[:-1, :, :], axis=0)  # Skip background layer
        hv_map = compute_hv_map(mask_1c)

        out = (
            torch.from_numpy(im),
            torch.from_numpy(mask),
            torch.from_numpy(hv_map),
        )

        return out