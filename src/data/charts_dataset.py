import os
import sys

from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import pdb

class ChartsDataset(Dataset):
    def __init__(self, args):
        super(ChartsDataset, self).__init__()
        dataroot = args.dataroot
        csvfile = args.csvfile
        imgdir = args.imgdir
        path_to_images = os.path.join(dataroot, imgdir)
        path_to_annotation_csv = os.path.join(dataroot, csvfile)
        imagefiles = os.listdir(path_to_images)
        f = lambda x: path_to_images + '/' + x
        self.train = args.train
        self.imagepaths = list(map(f, imagefiles))
        self.nSamples = len(imagefiles)
        transforms_list = [transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]
        
        self.transform = transforms.Compose(transforms_list)
        self.df = pd.read_csv(path_to_annotation_csv, 
            )

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imagepath = self.imagepaths[index]
        imagefile = os.path.basename(imagepath)
        image_index = imagefile.split('.')[0]
        img = Image.open(imagepath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        item = {'img': img, 'img_path': imagepath, 'idx':index}
        label = self.df.query('image_index==%s'%image_index)['type'].iloc[0]
        item['label'] = label
        return item
