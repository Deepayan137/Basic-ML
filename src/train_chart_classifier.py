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

from data.charts_dataset import ChartsDataset
from models.networks import ConvModel
from utils.utils import AverageMeter, LabelConverter, plot_losses
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, default='demo')
    parser.add_argument('--dataroot', type=str, default='../data/charts')
    parser.add_argument('--imgdir', type=str, default='train_val')
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
    parser.add_argument('--mode', type=str, default='py')
    parser.add_argument('--csvfile', type=str, default='train_val.csv')
    parser.add_argument('--norm', type=str, default='bn')
    parser.add_argument('--pretrained', action='store_true')
    args = parser.parse_args()

    args.n_classes = 5
    data = ChartsDataset(args)
    args.model = ConvModel(args)
    args.criterion = torch.nn.CrossEntropyLoss()
    args.optimizer = torch.optim.Adam(args.model.parameters(), lr=args.lr)
    from modules.chart_trainer import ChartsTrainer
    train_split = int(0.8*len(data))
    val_split = len(data) - train_split
    args.data_train, args.data_val = random_split(data, (train_split, val_split))
    args.imgdir = 'test'
    args.data_test = ChartsDataset(args)
    trainer = ChartsTrainer(args)
    trainer.setup(args)
    best_loss = np.Inf
    best_accuracy = 0.0
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    if args.train:
        for epoch in range(args.epochs):
            train_result = trainer.run_epoch()
            val_result = trainer.run_epoch(validation=True)
            print('Epochs: [%d]/[%d] Train Accuracy: %.4f Val Accuracy: %.4f'%(
                epoch+1, args.epochs, train_result['train_accuracy'],
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
        print('Best Accuracy on Val Dataset: %.4f at epoch: %d'%(
            best_accuracy, at_epoch))
        plot_losses(train_loss, val_loss, train_acc, val_acc, epoch, trainer.save_dir)
    else:
        args.train = False
        trainer.setup(args)
        pred_pairs = trainer.get_predictions()
        df = pd.DataFrame(pred_pairs, 
            columns=['img_path', 'prediction'])
        save_dir = trainer.save_dir
        # df.to_csv(os.path.join(save_dir, 'test_predictions.csv'))



        # python train_chart_classifier.py --epochs 100 --name charts_pretrained_with_scheduler --mode pytorch --csvfile train_val.csv  --pretrained  --train