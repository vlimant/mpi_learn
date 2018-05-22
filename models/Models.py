### Predefined Keras models

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
import keras.backend as K

def make_model(model_name):
    """Constructs the Keras model indicated by model_name"""
    model_maker_dict = {
            'example':make_example_model,
            'mnist':make_mnist_model,
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
