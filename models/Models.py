### Predefined Keras models

#import setGPU
#import numpy as np
import sys
try:
    from keras.models import Sequential, Model
    from keras.layers import Dense, Activation, Dropout, Flatten, Input, Permute
    from keras.layers import Convolution2D, MaxPooling2D, Conv2D
    import keras.backend as K
except:
    print ("no keras support")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except:
    print ("no torch support")

def model_function(model_name):
    """Constructs the Keras model indicated by model_name"""
    model_maker_dict = {
            'example':make_example_model,
            'mnist':make_mnist_model,
            'cifar10':make_cifar10_model,
            'mnist_torch':make_mnist_torch_model,
            'topclass': make_topclass_model,
            'topclass_torch':make_topclass_torch_model
        
            }
    return model_maker_dict[model_name]    
def make_model(model_name, **args):
    m_fn = model_function(model_name)
    if args and hasattr(m_fn,'parameter_range'):
        provided = set(args.keys())
        accepted = set([a.name for a in m_fn.parameter_range])
        if not provided.issubset( accepted ):
            print ("provided arguments",sorted(provided),"do not match the accepted ones",sorted(accepted))
            sys.exit(-1)
    return model_function(model_name)(**args)

def make_example_model():
    """Example model from keras documentation"""
    model = Sequential()
    model.add(Dense(output_dim=64, input_dim=100))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=10))
    model.add(Activation("softmax"))
    return model

def make_topclass_model(**args):
    if args:print ("receiving arguments",args)    
    conv_layers=args.get('conv_layers',2)
    dense_layers=args.get('dense_layers',2)
    dropout=args.get('dropout',0.2)
    kernel = args.get('kernel_size',3)
    classes=3
    in_channels=5
    in_ch = in_channels
    ## the trace in the input file is 750, 150, 94, 5
    input = Input( (150,94,in_ch))
    ## convs
    c = input
    for i in range(conv_layers):
        channel_in = in_ch*((i+1)%5)
        channel_out = in_ch*((i+2)%5)
        if channel_in == 0: channel_in += 1
        if channel_out == 0: channel_out += 1
        c = Conv2D( filters=channel_out, kernel_size=(kernel,kernel) , strides=1, padding="same", activation = 'relu') (c)
    c = Conv2D(1, (kernel,kernel), activation = 'relu',strides=2, padding="same")(c)

    ## pooling
    pool = args.get('pool', 10)
    m  = MaxPooling2D((pool,pool))(c)
    f = Flatten()(m)
    d = f
    base = args.get('hidden_factor',5)*100
    for i in range(dense_layers):
        N = int(base//(2**(i+1)))
        d = Dense( N, activation='relu')(d)
        if dropout:
            d = Dropout(dropout)(d)
    o = Dense(classes, activation='softmax')(d)

    model = Model(inputs=input, outputs=o)
    #model.summary()
    return model

def make_cifar10_model(**args):
    if args:print ("receiving arguments",args)    
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
    do4 = args.get('dropout1', 0.25)
    do5 = args.get('dropout2', 0.5)
    
    # tune the dense layers independently
    dense1 = args.get('dense1', 512)
    dense2 = args.get('dense2', 256)
    
    if K.image_dim_ordering() == 'th':
        input_shape = (3, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 3)

    #act = 'sigmoid'
    act = 'relu'
        
    i = Input( input_shape)
    l = Conv2D(nb_filters1,( ks, ks), padding='same', activation = act)(i)
    l = MaxPooling2D(pool_size=pool_size)(l)
    #l = Dropout(do1)(l)

    l = Conv2D(nb_filters2, (ks, ks), padding='same',activation=act)(l)
    #l = Conv2D(nb_filters2, (ks, ks))(l)
    l = MaxPooling2D(pool_size=pool_size)(l)
    #l = Dropout(do2)(l)

    l = Conv2D(nb_filters3, (ks, ks), padding='same',activation=act)(l)
    #l = Conv2D(nb_filters3, (ks, ks))(l)
    l = MaxPooling2D(pool_size=pool_size)(l)
    #l = Dropout(do3)(l)

    l = Flatten()(l)
    l = Dense(dense1,activation=act)(l)
    l = Dropout(do4)(l)
    l = Dense(dense2,activation=act)(l)
    l =Dropout(do5)(l)
    
    o = Dense(nb_classes, activation='softmax')(l)

    model = Model(inputs=i, outputs=o)
    #model.summary()
    
    return model

def make_mnist_model(**args):
    """MNIST ConvNet from keras/examples/mnist_cnn.py"""
    #np.random.seed(1337)  # for reproducibility
    if args:print ("receiving arguments",args)
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
    model.add(Convolution2D(nb_filters, (ks, ks),
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, (ks, ks)))
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

def make_mnist_torch_model(**args):
    if args:print ("receiving arguments",args)    
    from TorchModels import MNistNet
    model = MNistNet(**args)
    return model

def make_topclass_torch_model(**args):
    if args:print ("receiving arguments",args)    
    conv_layers=args.get('conv_layers',2)
    dense_layers=args.get('dense_layers',2)
    dropout=args.get('dropout',0.5)
    classes=3
    in_channels=5
    from TorchModels import CNN
    model = CNN(conv_layers=conv_layers, dense_layers=dense_layers, dropout=dropout, classes=classes, in_channels=in_channels)
    return model

try:
    from skopt.space import Real, Integer, Categorical
    make_mnist_model.parameter_range =     [
        Integer(10,50, name='nb_filters'),
        Integer(2,10, name='pool_size'),
        Integer(2,10, name='kernel_size'),
        Integer(50,200, name='dense'),
        Real(0.0, 1.0, name='dropout')
    ]
    make_mnist_torch_model.parameter_range = [
        Integer(2,10, name='kernel_size'),
        Integer(50,200, name='dense'),
        Real(0.0, 1.0, name='dropout')
    ]
    make_topclass_model.parameter_range =   [
        Integer(1,6, name='conv_layers'),
        Integer(1,6, name='dense_layers'),
        Integer(1,6, name='kernel_size'),
        Real(0.0, 1.0, name='dropout')
    ]
    make_topclass_torch_model.parameter_range =    [
        Integer(1,6, name='conv_layers'),
        Integer(1,6, name='dense_layers'),
        Real(0.0,1.0, name='dropout')
    ]
    make_cifar10_model.parameter_range = [
        Integer(10,300, name='nb_filters1'),
        Integer(10,300, name='nb_filters2'),
        Integer(10,300, name='nb_filters3'),
        Integer(50,1000, name='dense1'),
        Integer(50,1000, name='dense2'),
        Real(0.0, 1.0, name='dropout1'),
        Real(0.0, 1.0, name='dropout2')
    ]
except:
    pass

