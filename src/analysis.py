import os
import pdb
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
import torch
from data.digits_dataset import DigitsDataset
from data.charts_dataset import ChartsDataset
from data.utils import batchify
from utils.utils import OneHot, AverageMeter, plot_losses
from modules.digit_trainer import DigitTrainer
from modules.chart_trainer import ChartsTrainer
from collections import OrderedDict
from models.networks import ConvModel
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
    parser.add_argument('--csvfile', type=str, default='train_val.csv')
    parser.add_argument('--norm', type=str, default='bn')
    args = parser.parse_args()

    args.imgdir = 'test'
    args.data_train = None
    args.transform = None
    args.data_val = ChartsDataset(args)
    args.model = ConvModel(args)
    args.optimizer = None
    args.criterion = None
    args.layers_size = [512, 128, 10]
    trainer = ChartsTrainer(args)
    trainer.setup(args)
    loader = trainer.val_dataloader()
    triplets = []
    error_count = 0
    data_size = len(args.data_val)
    for i, batch in enumerate(loader):
        X, Y, img_path = batch['img'], batch['label'],\
            batch['img_path']
        Y = trainer.converter(Y)
        A, _ = trainer.forward(X)
        y_hat = np.argmax(A, axis=0)
        Y = np.argmax(Y, axis=1)
        err_indices = np.where(y_hat != Y)[0]
        batch_error_count = len(err_indices)
        error_count += batch_error_count
        if batch_error_count > 0:
            err_paths = [img_path[i] for i in err_indices]
            error_preds = y_hat[err_indices]
            exp_labels = Y[err_indices]
            units = list(zip(error_preds, exp_labels,
                err_paths))
            triplets.extend(units)
    accuracy = (1-(error_count/data_size))*100
    print('Accuracy: %.4f'%accuracy)
    df = pd.DataFrame(triplets, columns=['pred', 'truth', 'path'])
    save_dir = trainer.save_dir
    df.to_csv(os.path.join(save_dir, 'analysis.csv'))
