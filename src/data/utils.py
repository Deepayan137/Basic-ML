import pdb
import random
import numpy as np
from data.digits_dataset import DigitsDataset


def batchify(data=None, batch_size=None, shuffle=False):
    l = len(data)
    n = batch_size
    indices = np.arange(0, l)
    if shuffle:
        random.shuffle(indices)
    for start_idx in range(0, l - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield data[excerpt]

class Batcher:

    def __init__(self, data=None, batch_size=None, shuffle=False):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        return self
    def __len__(self):
        return len(self.data)//self.batch_size
    def __next__(self):
        l = len(self.data)
        n = self.batch_size
        indices = np.arange(0, l)
        if self.shuffle:
            random.shuffle(indices)
        for start_idx in range(0, l - self.batch_size + 1, self.batch_size):
            if self.shuffle:
                excerpt = indices[start_idx:start_idx + self.batch_size]
            else:
                excerpt = np.arange(slice(start_idx, start_idx + self.batch_size))
        batch = np.zeros((self.batch_size, 1024), dtype='float')
        labels, paths, idxs = [], [], []
        for i, index in enumerate(excerpt):
            img, label, path, idx = self.data[index]['img'], \
            self.data[index]['label'], self.data[index]['img_path'], self.data[index]['idx']
            batch[i, :] = img
            labels.append(label)
            paths.append(path)
            idxs.append(idx)
        return {'img':batch, 'label':labels, 'img_path': paths, 'index':idxs}

