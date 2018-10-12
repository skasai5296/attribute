import os
import random
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class DeepFashion(Dataset):
    '''
    DeepFashion dataset class
    img_dir : directory of image files
    ann_dir : directory of annotation
    transform : how to transform images, should be from torchvision.transforms
    '''
    def __init__(self, data_dir, ann_dir, mode='Train', transform=None):
        self.data_dir = data_dir
        self.imdir = []
        self.ann = []
        self.anndir = os.path.join(self.data_dir, ann_dir)
        self.transform = transform
        
        with open(self.anndir, 'r') as ann:
            for i, line in enumerate(ann):
                if i == 0:
                    self.len = int(line.rstrip('\n'))
                    taridx = set(range(self.len))
                    validx = set(random.sample(taridx, self.len // 5))
                    trainidx = taridx - validx
                elif i > 1:
                    if mode == 'Train':
                        self.idx = list(trainidx)
                    elif mode == 'Val':
                        self.idx = list(validx)
                    else:
                        raise NotImplementedError()
                    imdir, ann = line.rstrip('\n').split(maxsplit=1)
                    self.imdir.append(os.path.join(self.data_dir, imdir))
                    ann_np = np.array([int(i) for i in ann.split()])
                    ann_np[ann_np == -1] = 0
                    self.ann.append(ann_np)
        print('completed loading {} dataset'.format(mode), flush=True)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        imdirs = [self.imdir[i] for i in self.idx]
        impath = imdirs[idx]
        img = Image.open(impath)
        ann = self.ann[idx]
        
        if self.transform:
            img = self.transform(img)

        sample = {'image': img, 'attributes': ann}

        return sample

class CelebA(Dataset):
    '''
    CelebA dataset class
    data_dir : root directory
    img_dir : directory of image files
    ann_dir : directory of annotation
    transform : how to transform images, should be from torchvision.transforms
    '''
    def __init__(self, data_dir, img_dir, ann_dir, mode='Train', transform=None):
        self.data_dir = data_dir
        self.img_dir = os.path.join(self.data_dir, img_dir)
        self.imdir = []
        self.ann = []
        self.anndir = os.path.join(self.data_dir, ann_dir)
        self.transform = transform
        self.df = pd.read_csv(self.anndir).replace(-1, 0)
        if mode == 'Train':
            self.df = self.df.iloc[:len(self.df) // 5 * 4]
        elif mode == 'Test':
            self.df = self.df.iloc[len(self.df) // 5 * 4]

        print('completed loading {} dataset'.format(mode), flush=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        impath = os.path.join(self.img_dir, self.df.loc[idx, 'image_id'])
        img = Image.open(impath)
        ann = self.df.iloc[idx, 1:].values.astype(int)

        if self.transform:
            img = self.transform(img)

        sample = {'image': img, 'attributes': ann}

        return sample


def collate_fn(data):
    """
    converts list of samples to a batch.
    sample : {'image': (250, 250, 3), 'attributes': (1000,)}
    """
    image = torch.stack([sample['image'] for sample in data], 0)
    attributes = torch.Tensor([sample['attributes'] for sample in data])
    return {'image': image, 'attributes': attributes}


