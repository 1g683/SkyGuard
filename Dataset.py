import torch
from torch.utils.data import Dataset
import os
import random
import pandas as pd
import numpy as np
from PIL import Image
import config

class SatUAVDataset(Dataset):
    '''
    Raw images + augmented pairs.
    '''

    def __init__(self, csv_meta, csv_file, root_dir=config.DATA_DIR, transform=None,processor=None):
        self.meta = pd.read_csv(os.path.join(root_dir, csv_meta))
        self.path_list = list(self.meta.itertuples(index=False, name=None))
        self.file_frame = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.transform = transform
        self.raw_len = len(self.file_frame) # number of raw images as descried in csv_file
        self.processor = processor

    def __len__(self):
        return self.raw_len * len(self.path_list)

    @staticmethod
    def image_name(raw_name, aug_trick):
        #assert len(raw_name) == 10, "raw_name is :"+raw_name+", whose len is not 10."
        if aug_trick.lower() == 'raw':
            return raw_name
        l = raw_name.split(".")
        l[0] += "_"+aug_trick[0].lower()
        return ".".join(l)

    def __getitem__(self, idx):

        # construct A related information
        A = {
            'aug_trick': self.path_list[idx//self.raw_len][0],
            'dir': os.path.join(self.root_dir, self.path_list[idx//self.raw_len][1]),
            'idx': idx % self.raw_len,
        }
        A['raw_name'] = self.file_frame.iloc[A['idx'],0]
        A['path'] = os.path.join(A['dir'], self.image_name(A['raw_name'], A['aug_trick']))

        # pick random augmentation trick for B, if unpaired, pick a shift for B
        rand_trick_idx = np.random.randint(len(self.path_list))
        label = [0] # paired : 0, unpaired : 1
        if random.choice([True, False]): # if unpaired
            shift = np.random.randint(low=1, high=self.raw_len)
            label[0] = 1
        else:
            shift = 0

        # construct B related information
        B = {
            'aug_trick': self.path_list[rand_trick_idx][0],
            'dir': os.path.join(self.root_dir, self.path_list[rand_trick_idx][1]),
            'idx': (idx+shift) % self.raw_len,
        }
        B['raw_name'] = self.file_frame.iloc[B['idx'], 1]
        B['path'] = os.path.join(B['dir'], self.image_name(B['raw_name'], B['aug_trick']))

        A_img = Image.open(A['path']).convert('RGB')
        B_img = Image.open(B['path']).convert('RGB')
        if self.processor:
            A_img = self.processor(images=A_img, return_tensors="pt")
            A_img = A_img['pixel_values'].squeeze(0)
            B_img = self.processor(images=B_img, return_tensors="pt")
            B_img = B_img['pixel_values'].squeeze(0)
        if self.transform:
            A_img = self.transform(A_img)
            B_img = self.transform(B_img)
        sample = {'A': A_img, 'B': B_img, 'label': torch.FloatTensor(label)}

        return sample