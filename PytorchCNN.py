import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

class MNistNet(torch.nn.Module):
    def __init__(self, **args):
        super(MNistNet, self).__init__()
        ks = args.get('kernel_size',5)
        do = args.get('dropout',0.5)
        dense = args.get('dense',50)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=ks)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=ks)
        self.conv2_drop = nn.Dropout2d(do)
        self.fc1 = nn.Linear(320, dense)
        self.fc2 = nn.Linear(dense, 10)
        
    def forward(self, x):
        x = x.permute(0,3,1,2).float()
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        #return F.log_softmax(x, dim=1)
        #return F.softmax(x)
        #return F.cross_entropy(x)
        return x
        
            

### Build a customized CNN with given hyperparameters
class _ConvBlock(nn.Sequential):
    def __init__(self, conv_layers, dropout, in_ch=5):
        super().__init__()
        for i in range(conv_layers):
            channel_in = in_ch*((i+1)%5)
            channel_out = in_ch*((i+2)%5)
            if channel_in == 0: channel_in += 1
            if channel_out == 0: channel_out += 1
            self.add_module('convlayer%d'%(i), nn.Conv2d(channel_in, out_channels=channel_out,kernel_size=(3,3),stride=1, padding=1))
            self.add_module('relu%d'%(i), nn.ReLU(inplace=True))
        self.add_module('convlayer%d'%(conv_layers), nn.Conv2d(channel_out, out_channels=1, kernel_size=(3,3), stride=2, padding=1))
        self.dropout = dropout

    def forward(self, x):
        x = super().forward(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class _DenseBlock(nn.Sequential):
    def __init__(self, dense_layers, dropout ,base):
        super().__init__()
        for i in range(dense_layers):
            il = int(base//(2**i))
            ol = int(base//(2**(i+1)))
            print (il,"=>",ol)
            self.add_module('denselayer%d'%(i), nn.Linear(il, ol))
            self.add_module('relu%d'%(i), nn.ReLU(inplace=True))
        self.dropout = dropout

    def forward(self, x):
        x = super().forward(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class CNN(nn.Module):
    def __init__(self, conv_layers=2, dense_layers=2, dropout=0.5, classes=3, in_channels=5):
        super().__init__()
        self.build_net(conv_layers, dense_layers, dropout, classes, in_channels)
    
    def build_net(self,*args, **kwargs):
        base_2 = 10
        base = base_2**2
        self.conv_layers = _ConvBlock(args[0], args[2], args[4])
        self.dense_layers = _DenseBlock(args[1], args[2], base)
        self.adapt_pool = nn.AdaptiveMaxPool2d((base_2,base_2))
        il = int(base//(2**(args[1])))
        ol = int(args[3])
        print (il,"=>",ol)
        self.output = nn.Linear(il, ol)

    def forward(self, x):
        x = x.permute(0,3,1,2).float()        
        x = self.conv_layers(x)
        x = self.adapt_pool(x)
        x = x.view(x.shape[0], -1) # flatten
        x = self.dense_layers(x)
        return self.output(x)
            

