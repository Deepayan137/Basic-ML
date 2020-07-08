import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as f


class Sigmoid:
    def __call__(self, x):
        return 1/(1+np.exp(-x))
class ReLU:
    def __call__(self, x):
        return np.maximum(0, x)
class Tanh():
    def __call__(self, x):
        return np.tanh(x)

class Step():
    def __call__(self, x):
        f = lambda z: 1 if z>=0 else 0
        x = list(map(f, x))
        return np.array(x)

class Perceptron():
        def __init__(self, input_dim, output_dim, 
            activation='none', use_bias=False):
            super(Perceptron, self).__init__()
            
            if use_bias:
                input_dim += 1
            self.input_dim = input_dim
            self.output_dim = output_dim

            if activation == 'step':
                self.activation = Step()
            elif activation == 'none':
                self.activation = None
            else:
                activation
                assert 0, "Unsupported activation: {}".format(activ)
            self._init_weights()
        
        def _init_weights(self):
            self.weights = np.ones((self.input_dim, self.output_dim), dtype='float')
            
        def __call__(self, input):
            output = np.dot(input, self.weights)
            if self.activation:
                return self.activation(output)
            return output


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride=1,
                 padding=1, norm='none', activation='relu'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)
    
    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out
import torchvision.models as models
class ConvModel(nn.Module):
    def __init__(self, args):
        super(ConvModel, self).__init__()
        input_dim = args.input_dim
        n_classes = args.n_classes
        dim = args.dim
        n_layers = args.n_layers
        self.layers = []
        self.args = args
        if not args.pretrained:
            
            self.layers += [Conv2dBlock(input_dim, dim, 3, activation='relu')]
            self.layers += [nn.MaxPool2d(2,2)]
            for _ in range(n_layers-1):
                self.layers += [Conv2dBlock(dim, dim * 2, 3, norm=args.norm, activation='relu')]
                self.layers += [nn.MaxPool2d(2,2)]
                dim *= 2
            self.layers = nn.Sequential(*self.layers)
            self.fc = LinearBlock(dim*32*32, n_classes, activation='none')
            self.output_dim = dim
        else:   
            vgg16 = models.vgg16(pretrained=True).cuda()
            # vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
            print(vgg16.classifier[6].out_features) # 1000 
            # Freeze training for all layers
            for param in vgg16.features.parameters():
                param.require_grad = False
            
            self.features = nn.Sequential(*list(vgg16.features.children()))
            self.fc1 = LinearBlock(8192, 4096, activation='relu')
            self.dropout1 = nn.Dropout(p=0.5)
            self.fc2 = LinearBlock(4096, 2048, activation='relu')
            self.dropout2 = nn.Dropout(p=0.5)
            self.fc_out = LinearBlock(2048, n_classes, activation='none')
            
        
    def forward(self, x):
        if not self.args.pretrained:
            features = self.layers(x)
            N, C, H, W = features.size()
            out = self.fc(features.view(N, -1))
        else:
            features = self.features(x)
            N, C, H, W = features.size()
            features = self.fc1(features.view(N, -1))
            features = self.dropout1(features)
            features = self.fc2(features)
            features = self.dropout2(features)
            out = self.fc_out(features)
        return out

# from argparse import ArgumentParser
# from tqdm import tqdm
# parser = ArgumentParser()
# args = parser.parse_args()
# args.input_dim = 4
# args.n_classes = 5
# args.n_layers = 2
# args.dim = 64
# model = ConvModel(args)
# print(model)


