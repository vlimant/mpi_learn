### Predefined Keras models

import numpy as np
try:
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout, Flatten
    from keras.layers import Convolution2D, MaxPooling2D
    import keras.backend as K
except:
    print ("no keras support")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except:
    print ("no torch support")

def make_model(model_name):
    """Constructs the Keras model indicated by model_name"""
    model_maker_dict = {
            'example':make_example_model,
            'mnist':make_mnist_model,
            'cifar10':make_cifar10_model,
            'mnist_torch':make_mnist_torch_model,
            'topclass_torch':make_topclass_torch_model
            }
    return model_maker_dict[model_name]()

def make_example_model():
    """Example model from keras documentation"""
    model = Sequential()
    model.add(Dense(output_dim=64, input_dim=100))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=10))
    model.add(Activation("softmax"))
    return model

def make_cifar10_model(**args):
    nb_classes = 10
    img_rows, img_cols = 32, 32
    
    # use 1 kernel size for all convolutional layers
    ks = args.get('kernel_size', 3)
    
    # tune the number of filters for each convolution layer
    nb_filters1 = args.get('nb_filters1', 48)
    nb_filters2 = args.get('nb_filters2', 96)
    nb_filters3 = args.get('nb_filters3', 192)
    
    # tune the pool size once
    ps = args.get('pool_size', 2)
    pool_size = (ps,ps)
    
    # tune the dropout rates independently
    do1 = args.get('dropout1', 0.25)
    do2 = args.get('dropout2', 0.25)
    do3 = args.get('dropout3', 0.25)
    do4 = args.get('dropout4', 0.25)
    do5 = args.get('dropout5', 0.5)
    
    # tune the dense layers independently
    dense1 = args.get('dense1', 512)
    dense2 = args.get('dense2', 256)
    
    if K.image_dim_ordering() == 'th':
        input_shape = (3, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 3)
    
    model = Sequential()
    model.add(Convolution2D(nb_filters1, ks, ks,
                            border_mode='same',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters1, ks, ks))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(do1))
    
    model.add(Convolution2D(nb_filters2, ks, ks, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters2, ks, ks))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(do2))
    
    model.add(Convolution2D(nb_filters3, ks, ks, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters3, ks, ks))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(do3))
    
    model.add(Flatten())
    model.add(Dense(dense1))
    model.add(Activation('relu'))
    model.add(Dropout(do4))
    model.add(Dense(dense2))
    model.add(Activation('relu'))
    model.add(Dropout(do5))
    
    model.add(Dense(nb_classes, activation='softmax'))
    
    return model

def make_mnist_model(**args):
    """MNIST ConvNet from keras/examples/mnist_cnn.py"""
    np.random.seed(1337)  # for reproducibility
    nb_classes = 10
    # input image dimensions
    img_rows, img_cols = 28, 28
    # number of convolutional filters to use
    nb_filters = args.get('nb_filters',32)
    # size of pooling area for max pooling
    ps = args.get('pool_size',2)
    
    # convolution kernel size
    ks = args.get('kernel_size',3)
    do = args.get('dropout', 0.25)
    dense = args.get('dense', 128)
    
    pool_size = (ps,ps)
    if K.image_dim_ordering() == 'th':
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)
    model = Sequential()
    model.add(Convolution2D(nb_filters, ks, ks,
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, ks, ks))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(do))
    model.add(Flatten())
    model.add(Dense(dense))
    model.add(Activation('relu'))
    model.add(Dropout(do))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    return model

class MNistNet(torch.nn.Module):
    def __init__(self):
        super(MNistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = x.permute(0,3,1,2).float()
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        #return F.softmax(x)
        #return F.cross_entropy(x)
        #return x
        
            
def make_mnist_torch_model(**args):
    model = MNistNet()
    return model

def make_topclass_torch_model(**args):
    conv_layers=2
    dense_layers=2
    dropout=0.5
    classes=3
    in_channels=5
    from PytorchCNN import CNN
    model = CNN(conv_layers=conv_layers, dense_layers=dense_layers, dropout=dropout, classes=classes, in_channels=in_channels)
    return model
