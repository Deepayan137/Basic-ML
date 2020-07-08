import os
import pdb 
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.nn.utils.clip_grad import clip_grad_norm_

from data.charts_dataset import ChartsDataset
from models.networks import ConvModel
from utils.utils import AverageMeter, LabelConverter, plot_losses
from .base_trainer import BaseTrainer

class ChartsTrainer(BaseTrainer):
    def __init__(self, args):
        BaseTrainer.__init__(self, args)
        self.train = args.train
        self.device = torch.device("cuda:0" if 
            torch.cuda.is_available() else "cpu")
        self.model = args.model.to(self.device)
        self.optimizer = args.optimizer
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=args.epochs)
        self.criterion  = args.criterion
        self.data_train = args.data_train
        self.data_val = args.data_val
        self.data_test = args.data_test
        self.batch_size = args.batch_size
        self.converter = LabelConverter()
        self.init_meters()

    def init_meters(self):
        self.avgTrainLoss = AverageMeter("Train loss")
        self.avgTrainAccuracy = AverageMeter("Train Accuracy")
        self.avgValLoss = AverageMeter("Val loss")
        self.avgValAccuracy = AverageMeter("Val Accuracy")

    def forward(self, batch, validation=False):
        input_, labels = batch['img'].to(self.device),\
                        batch['label']
        bs = len(labels)
        labels = self.converter.encode(labels)
        logits = self.model(input_)
        logits = logits.contiguous().cpu()
        loss = self.criterion(logits, labels)
        _, predictions = logits.max(1)
        accuracy = 0.0
        correct = torch.sum(predictions==labels).item()
        accuracy = (correct/bs)*100
        return loss, accuracy, predictions

    def step(self):
        self.max_grad_norm = 0.05
        clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def get_predictions(self):
        loader = self.val_dataloader()
        desc = 'testing'
        self.model.eval()
        pbar = tqdm(loader, desc=desc)
        triplets = []
        data_size = len(self.data_val)
        error_count = 0
        for batch_nb, batch in enumerate(pbar):
            input_ = batch['img'].to(self.device)
            image_path = batch['img_path']
            logits = self.model(input_)
            logits = logits.contiguous().cpu()
            logits = torch.nn.functional.log_softmax(logits)
            
            _, y_hat = logits.max(1)
            Y = self.converter.encode(batch['label'])
            err_indices = np.where(y_hat.numpy() != Y.numpy())[0]
            batch_error_count = len(err_indices)
            if batch_error_count > 0:
                error_count += batch_error_count
                err_paths = [image_path[i] for i in err_indices]
                error_preds = y_hat[err_indices]
                exp_labels = Y[err_indices]
                units = list(zip(error_preds, exp_labels,
                    err_paths))
                triplets.extend(units)
        
        accuracy = (1-(error_count/data_size))*100
        print('Accuracy: %.4f'%accuracy)
        df = pd.DataFrame(triplets, columns=['pred', 'truth', 'path'])
        save_dir = self.save_dir
        df.to_csv(os.path.join(save_dir, 'analysis.csv'))
                
    

    def train_step(self, batch):
        self.model.train()
        loss, accuracy, _ = self.forward(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.step()
        output = OrderedDict({
            'loss': loss.item(),
            'accuracy':accuracy
            })
        return output

    def val_step(self, batch):
        self.model.eval()
        loss, accuracy, predictions = self.forward(batch)
        output = OrderedDict({
            'val_loss': loss.item(),
            'val_accuracy': accuracy
            })
        return output

    def train_end(self, outputs):
        for output in outputs:
            self.avgTrainLoss.add(output['loss'])
            self.avgTrainAccuracy.add(output['accuracy'])
        train_loss_mean = self.avgTrainLoss.compute()
        train_accuracy_mean = self.avgTrainAccuracy.compute()
        result = {'train_loss': train_loss_mean, 
        'train_accuracy': train_accuracy_mean}
        return result

    def val_end(self, outputs):
        for output in outputs:
            self.avgValLoss.add(output['val_loss'])
            self.avgValAccuracy.add(output['val_accuracy'])
        val_loss_mean = self.avgValLoss.compute()
        val_accuracy_mean = self.avgValAccuracy.compute()
        result = {'val_loss': val_loss_mean, 
        'val_accuracy':val_accuracy_mean}
        return result
    
    def train_dataloader(self):
        # logging.info('training data loader called')
        loader = torch.utils.data.DataLoader(self.data_train,
                batch_size=self.batch_size,
                shuffle=True,
                )
        return loader
        
    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(self.data_val,
                batch_size=self.batch_size,
                )
        return loader
    
    def test_dataloader(self):
        loader = torch.utils.data.DataLoader(self.data_test,
                batch_size=8
                )
        return loader

