import pdb
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
import torch
from data.digits_dataset import DigitsDataset
from data.utils import batchify
from utils.utils import OneHot, AverageMeter, plot_losses
from .base_trainer import BaseTrainer
from collections import OrderedDict

class DigitTrainer(BaseTrainer):
    def __init__(self, args):
        BaseTrainer.__init__(self, args)
        self.layers_size = args.layers_size
        self.parameters = {}
        self.L = len(self.layers_size)
        self.n = 0
        self.costs = []
        self.lr = args.lr
        self.batch_size = args.batch_size
        np.random.seed(1)
        self.data_train = args.data_train
        self.data_val = args.data_val
        self.layers_size.insert(0, 1024)
        self.converter = OneHot(args.n_classes)
        self.gradients = {}
        self.initialize_parameters()
        self.init_meters()

    def init_meters(self):
        self.avgTrainLoss = AverageMeter("Train loss")
        self.avgTrainAccuracy = AverageMeter("Train Accuracy")
        self.avgValLoss = AverageMeter("Val loss")
        self.avgValAccuracy = AverageMeter("Val Accuracy")

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
 
    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)
 
    def initialize_parameters(self):
        np.random.seed(1)
        for l in range(1, len(self.layers_size)):
            self.parameters["W" + str(l)] = np.random.randn(self.layers_size[l], self.layers_size[l - 1]) / np.sqrt(
                self.layers_size[l - 1])
            self.parameters["b" + str(l)] = np.zeros((self.layers_size[l], 1))
 
    def forward(self, X):
        store = {}
        A = X.T
        for l in range(self.L - 1):
            Z = self.parameters["W" + str(l + 1)].dot(A) + self.parameters["b" + str(l + 1)]
            A = self.sigmoid(Z)
            store["A" + str(l + 1)] = A
            store["W" + str(l + 1)] = self.parameters["W" + str(l + 1)]
            store["Z" + str(l + 1)] = Z
 
        Z = self.parameters["W" + str(self.L)].dot(A) + self.parameters["b" + str(self.L)]
        A = self.softmax(Z)
        store["A" + str(self.L)] = A
        store["W" + str(self.L)] = self.parameters["W" + str(self.L)]
        store["Z" + str(self.L)] = Z
        return A, store
 
    def sigmoid_derivative(self, Z):
        s = 1 / (1 + np.exp(-Z))
        return s * (1 - s)
 
    def backward(self, X, Y, store):
        derivatives = {}
 
        store["A0"] = X.T
 
        A = store["A" + str(self.L)]
        dZ = A - Y.T
 
        dW = dZ.dot(store["A" + str(self.L - 1)].T)/ len(Y) 
        db = np.sum(dZ, axis=1, keepdims=True)/ len(Y)
        dAPrev = store["W" + str(self.L)].T.dot(dZ)
 
        derivatives["dW" + str(self.L)] = dW
        derivatives["db" + str(self.L)] = db
 
        for l in range(self.L - 1, 0, -1):
            dZ = dAPrev * self.sigmoid_derivative(store["Z" + str(l)])
            dW = 1. / len(Y) * dZ.dot(store["A" + str(l - 1)].T)
            db = 1. / len(Y) * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                dAPrev = store["W" + str(l)].T.dot(dZ)
 
            derivatives["dW" + str(l)] = dW
            derivatives["db" + str(l)] = db
 
        return derivatives

    def step(self):
        for l in range(1, self.L + 1):
            self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - self.lr * self.gradients[
                "dW" + str(l)]
            self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - self.lr * self.gradients[
                "db" + str(l)]

    def train_step(self, batch):    
        X, Y = batch['img'], batch['label']
        Y = self.converter(Y)
        A, store = self.forward(X)
        loss = -np.mean(np.multiply(Y, np.log(A.T+ 1e-8)))
        accuracy = self.predict(A, Y)
        self.gradients_prev = {k:0.9*v for k,v in self.gradients.items()}
        self.gradients = self.backward(X, Y, store)
        self.gradients = {k: self.gradients.get(k, 0) + self.gradients_prev.get(k, 0) for k in self.gradients.keys() | self.gradients_prev.keys()}
        self.step()
        output = OrderedDict({
            'loss': loss,
            'accuracy':accuracy
            })
        return output

    def train_end(self, outputs):
        for output in outputs:
            self.avgTrainLoss.add(output['loss'])
            self.avgTrainAccuracy.add(output['accuracy'])
        result = {'train_loss': self.avgTrainLoss.compute(), 
        'train_accuracy': self.avgTrainAccuracy.compute()}
        return result

    def val_step(self, batch):       
        X, Y = batch['img'], batch['label']
        Y = self.converter(Y)
        A, store = self.forward(X)
        loss = -np.mean(np.multiply(Y, np.log(A.T+ 1e-8)))
        accuracy = self.predict(A, Y)
        output = OrderedDict({
            'val_loss': loss,
            'val_accuracy':accuracy
            })
        return output

    def val_end(self, outputs):
        for output in outputs:
            self.avgValLoss.add(output['val_loss'])
            self.avgValAccuracy.add(output['val_accuracy'])
        result = {'val_loss': self.avgValLoss.compute(), 
        'val_accuracy': self.avgValAccuracy.compute()}
        return result

    def train_dataloader(self):
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
        
    def predict(self, logits, Y):
        y_hat = np.argmax(logits, axis=0)
        Y = np.argmax(Y, axis=1)
        accuracy = (y_hat == Y).mean()
        return accuracy * 100
