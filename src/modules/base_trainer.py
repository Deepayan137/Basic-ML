import os
import logging
import numpy as np
from abc import ABC, abstractmethod
from collections import OrderedDict
from argparse import ArgumentParser
from tqdm import *
import pickle

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseTrainer(ABC):
    def __init__(self, args):

        self.args = args
        
        self.save_dir = os.path.join(args.checkpoints_dir, args.name)
        self.gmkdir(self.save_dir)
    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def train_dataloader(self):
        pass

    @abstractmethod
    def train_step(self, batch):
        pass

    @abstractmethod
    def train_end(self, outputs):
        pass

    def setup(self, args):
        if not args.train or args.continue_train:
            load_suffix = 'latest'
            self.load_networks()

    def gmkdir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    
    def run_epoch(self, validation=False):
        if not validation:
            loader = self.train_dataloader()
            desc = 'training'
        else:
            loader = self.val_dataloader()
            desc = 'validating'

        pbar = tqdm(loader, desc=desc)
        outputs = []
        for batch_nb, batch in enumerate(pbar):
            if not validation:
                output = self.train_step(batch)
            else:
                output = self.val_step(batch)
            pbar.set_postfix(output)
            outputs.append(output)
        if not validation:
            result = self.train_end(outputs)
        else:
            result = self.val_end(outputs)
        return result

    def load_networks(self):
        if self.args.mode == 'pytorch': 
            load_path = os.path.join(self.save_dir, 'latest.pth')
            print('loading the model from %s' % load_path)
            state_dict = torch.load(load_path, map_location=str(self.device))
            self.model.load_state_dict(state_dict)
        elif self.args.mode == 'numpy':
            load_path = os.path.join(self.save_dir, 'latest.pickle')
            print('loading the model from %s' % load_path)
            with open(load_path, 'rb') as f:
                self.parameters = pickle.load(f)


        
    def save_networks(self):
        if self.args.mode == 'pytorch':
            save_filename = 'latest.pth'
            save_path = os.path.join(self.save_dir, save_filename)
            net = self.model
            torch.save(net.state_dict(), save_path)
        elif self.args.mode == 'numpy':
            save_filename = 'latest.pickle'
            save_path = os.path.join(self.save_dir, save_filename)
            with open(save_path, 'wb') as f:
                pickle.dump(self.parameters, f)
            # np.save(save_path, self.parameters, allow_pickle=True)
