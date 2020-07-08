import os
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

from models.networks import Perceptron
from data.perceptron_dataset import Dataset, batchify
from utils.utils import L1Loss, plot_line
from modules.perceptron_trainer import Trainer


if  __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=int, default=0.1)
    parser.add_argument('--checkpoints_dir', type=str, default='../saves')
    args = parser.parse_args()

    input = np.array([[1, 1], [-1, -1], [0, 0.5], [0.1, 0.5], [0.2, 0.2], [0.9, 0.5]])
    targets = np.array([1, -1, -1, -1, 1, 1])
    args.train_data = Dataset(input, targets)
    args.val_data = None
    args.mode = 'numpy'
    trainer = Trainer(args)

    for i, epoch in enumerate(range(1, args.epochs)):
        result = trainer.run_epoch()
        filename = os.path.join(trainer.save_dir, 'plot_%d.png'%(i+1))
        plot_line(trainer.weights, filename)
        print("Epochs: [%d]/[%d]"%(epoch, args.epochs))
        error_count = result['error_count']
        if error_count == 0:
            print('No error')
            print(trainer.weights)
            break
            


