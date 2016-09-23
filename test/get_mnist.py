### This script downloads the MNIST dataset, unpacks it, splits it into four pieces, and saves 
# each piece in a separate h5 file.

from numpy import array_split
from keras.datasets import mnist
from keras.utils import np_utils
from keras import backend as K
import h5py

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

img_rows = 28
img_cols = 28
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

num_train_pieces = 8
num_test_pieces = 1
split_X_train = [ X.astype('float32') / 255 for X in array_split(X_train, num_train_pieces) ]
split_Y_train = [ np_utils.to_categorical(Y,10) for Y in array_split(Y_train, num_train_pieces) ]
split_X_test = [ X.astype('float32') / 255 for X in array_split(X_test, num_test_pieces) ]
split_Y_test = [ np_utils.to_categorical(Y,10) for Y in array_split(Y_test, num_test_pieces) ]

for i in range(num_train_pieces):
    train_outfile = h5py.File( "mnist_train_%d.h5" % i, 'w' )
    train_outfile.create_dataset( "features", data=split_X_train[i] )
    train_outfile.create_dataset( "labels", data=split_Y_train[i] )
    train_outfile.close()

for i in range(num_test_pieces):
    test_outfile = h5py.File( "mnist_test_%d.h5" % i, 'w' )
    test_outfile.create_dataset( "features", data=split_X_test[i] )
    test_outfile.create_dataset( "labels", data=split_Y_test[i] )
    test_outfile.close()
