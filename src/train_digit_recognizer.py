import pdb
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
import torch
from data.digits_dataset import DigitsDataset
from data.utils import batchify
from utils.utils import OneHot, AverageMeter, plot_losses
from modules.digit_trainer import DigitTrainer
from collections import OrderedDict
from data.transforms import Gaussian, RandomFlip, composite_function
if __name__ == '__main__':
 
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, default='demo')
    parser.add_argument('--dataroot', type=str, default='../data/digits')
    parser.add_argument('--imgdir', type=str, default='train')
    parser.add_argument('--checkpoints_dir', type=str, default='../saves')
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--continue_train', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--mode', type=str, default='numpy')

    args = parser.parse_args()

    # args.transform = composite_function(RandomFlip(0.5))
    args.transform = None
    args.data_train = DigitsDataset(args)
    args.imgdir = 'val'
    args.transform = None
    args.data_val = DigitsDataset(args)
    args.layers_size = [512, 10]
    trainer = DigitTrainer(args)
    trainer.setup(args)
    best_loss = np.Inf
    best_accuracy = 0.0
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    for epoch in range(args.epochs):
        train_result = trainer.run_epoch()
        val_result = trainer.run_epoch(validation=True)
        print('Epochs: [%d]/[%d] Train Accuracy: %.4f Val Accuracy: %.4f'%(
            epoch, args.epochs, train_result['train_accuracy'],
            val_result['val_accuracy']))
        train_loss.append(train_result['train_loss'])
        val_loss.append(val_result['val_loss'])
        train_acc.append(train_result['train_accuracy'])
        val_acc.append(val_result['val_accuracy'])
        
        if val_result['val_accuracy'] > best_accuracy:
                best_accuracy = val_result['val_accuracy']
                at_epoch = epoch+1
            
        if val_result['val_loss'] < best_loss:
            print('Saving a new Model')
            trainer.save_networks()
        else:
            print('Old Loss: %.4f --> New Loss: %.4f'%(best_loss, val_loss))
    plot_losses(train_loss, val_loss, train_acc, val_acc, epoch, trainer.save_dir)
    print('Best Accuracy on Val Dataset: %.4f at epoch: %d'%(
        best_accuracy, at_epoch))
