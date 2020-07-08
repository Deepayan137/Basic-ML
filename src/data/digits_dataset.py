import os
import glob
import sys

from PIL import Image
import numpy as np
from numpy import asarray
from numpy import clip
import pandas as pd
import pdb

class DigitsDataset(object):
    def __init__(self, args):
        dataroot = args.dataroot
        imgdir = args.imgdir
        path = os.path.join(dataroot, imgdir)
        extensions = ['bmp', 'tiff']
        self.imagepaths = []
        for extension in extensions:
            self.imagepaths += glob.glob(os.path.join(path, '**', '*.' + extension), 
                recursive=True)
        self.nSamples = len(self.imagepaths)
        self.transform = args.transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        imagepath = self.imagepaths[index]
        imagefile = os.path.basename(imagepath)
        img = Image.open(imagepath).convert('L')
        pixels = asarray(img)
        pixels = pixels.astype('float32')
        mean, std = pixels.mean(), pixels.std()
        pixels = (pixels - mean) / std
        if self.transform:
            pixels = self.transform(pixels)
            pixels = np.asarray(pixels, dtype='float')
        img = clip(pixels, -1.0, 1.0)
        item = {'img': img.flatten(), 'img_path': imagepath, 'idx':index}
        item['label'] = int(imagepath.split('/')[-2])
        return item

