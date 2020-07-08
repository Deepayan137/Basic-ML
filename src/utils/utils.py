from matplotlib import pyplot as plt
plt.switch_backend('agg') # for servers not supporting display
import os
import pdb

def plot_line(w, filename):
    pos = np.array([[1.0, 0.2, 0.9], [1, 0.2, 0.5]])
    neg = np.array([[-1, 0.0, 0.1], [-1, 0.5, 0.5]])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.scatter(pos[0], pos[1], color='b')
    ax.scatter(neg[0], neg[1], color='r')
    xmin, xmax = plt.xlim() 
    xx = np.linspace(xmin, xmax)

    a = -w[0] / w[1]
    yy = a * xx
    ax.plot(xx, yy, 'k-')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.plot()
    plt.savefig(filename, dpi=100)
    plt.show()
    plt.close()

class AverageMeter:
    def __init__(self, name):
        self.name = name
        self.count = 0
        self.total = 0
        self.max = -1 * float("inf")
        self.min = float("inf")

    def add(self, element):
        self.total += element
        self.count += 1
        self.max = max(self.max, element)
        self.min = min(self.min, element)

    def compute(self):
        # pdb.set_trace()
        if self.count == 0:
            return float("inf")
        return self.total / self.count

    def __str__(self):
        return "%s (min, avg, max): (%.3lf, %.3lf, %.3lf)" % \
        (self.name, self.min, self.compute(), self.max)

class L1Loss:
    def __call__(self, predictions, targets):
        return (targets - predictions)

import torch

class LabelConverter(object):
    def __init__(self):
        classes = ['line', 'dot_line', 
                    'hbar_categorical', 'vbar_categorical', 
                    'pie']
        self.label2int = {}
        for i, cl in enumerate(classes):
            self.label2int[cl] = i
        self.int2label = {v:k for k,v in self.label2int.items()}

    def encode(self, labels):
        result = []
        for label in labels:
            result.append(self.label2int[label])
        labels = result
        return torch.LongTensor(labels)

    def decode(self, indices):
        result = []
        for index in indices:
            result.append(self.int2label[index.item()])
        return result

def plot_losses(train_loss, val_loss, train_acc, val_acc, epoch, save_dir):

    fig = plt.figure(figsize=(16,16))
    fig.suptitle('loss and accuracy', fontsize=20)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.title.set_text('Train-val loss VS #epochs')
    ax1.set_xlabel('#epochs')
    ax1.set_ylabel('Loss')
    ax1.plot(train_loss, color='r', label='train_loss')
    ax1.plot(val_loss, color='b', label='val loss')
    ax1.legend()
    
    ax2.title.set_text('Train-val accuracy VS #epochs')
    ax2.set_xlabel('#epochs')
    ax2.set_ylabel('Accuracy')
    ax2.plot(train_acc, color='r', label='train_accuracy')
    ax2.plot(val_acc, color='b', label='val_accuracy')
    ax2.legend()
    plt.savefig(os.path.join(save_dir,'losses.png'), dpi=100)
    plt.show()
    plt.close()
import numpy as np

class OneHot(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
    def __call__(self, labels):
        one_hot = np.zeros((len(labels), self.n_classes),dtype='float')
        for i in range(len(labels)):
            one_hot[i, labels[i]] = 1
        return one_hot
