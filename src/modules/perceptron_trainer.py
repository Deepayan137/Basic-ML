import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from collections import OrderedDict
from utils.utils import AverageMeter
from data.perceptron_dataset import batchify 
import pdb
from .base_trainer import BaseTrainer

class Trainer(BaseTrainer):
    def __init__(self, args):
        BaseTrainer.__init__(self, args)
        self.lr = args.lr
        self.train_data = args.train_data
        self.val_data = args.val_data
        self.batch_size = args.batch_size
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.init_meters()
        self.initialize_parameters()
    
    def init_meters(self):
        self.avgTrainLoss = AverageMeter("Train loss")

    def initialize_parameters(self):
        self.weights = np.ones((self.input_dim, self.output_dim), 
            dtype='float')
    
    def forward(self, batch):
        input, target = batch
        logits = (input).dot(self.weights)
        logits = np.multiply(target, logits.squeeze())
        return logits

    def update(self):
        self.w_grad = np.reshape(self.w_grad, 
            self.weights.shape)
        self.weights += self.lr * self.w_grad

    def train_step(self, batch):
        logits = self.forward(batch)
        loss = -np.sum(logits[np.where(logits<0)])
        error = logits[np.where(logits<0)].shape[0]
        X_err = batch[0][np.where(logits<0)[0]]
        y_err = batch[1][np.where(logits<0)[0]]
        if error > 0:
            self.w_grad = np.sum(np.multiply(y_err, X_err), axis=0)
            self.update()
        output = OrderedDict({
            'loss': loss,
            'error': error
            })
        return output


    def train_end(self, outputs):
        error_count = 0
        for output in outputs:
            self.avgTrainLoss.add(output['loss'])
            error_count+=output['error']

        train_loss_mean = self.avgTrainLoss.compute()
        result = {'train_loss': train_loss_mean,
        'error_count': error_count}
        return result

    def train_dataloader(self):
        return batchify(data=self.train_data,
                        batch_size=self.batch_size,
                        shuffle=True)
