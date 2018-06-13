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
            'mnist_torch':make_mnist_torch_model,
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
    pool_size = (ps,ps)
    # convolution kernel size
    ks = args.get('kernel_size',3)
    kernel_size = (ks, ks)
    do = args.get('drop_out', 0.25)
    dense = args.get('dense', 128)
    if K.image_dim_ordering() == 'th':
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)
    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
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
    
