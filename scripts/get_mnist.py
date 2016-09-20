### This script downloads the MNIST dataset, unpacks it, splits it into four pieces, and saves 
# each piece in a separate h5 file.

from numpy import array_split
from keras.datasets import mnist
import h5py

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

num_pieces = 4
split_X_train = array_split(X_train, num_pieces)
split_Y_train = array_split(Y_train, num_pieces)

for i in range(num_pieces):
    outfile = h5py.File( "/data/duanders/mnist_train_%d.h5" % i, 'w' )
    outfile.create_dataset( "features", data=split_X_train[i] )
    outfile.create_dataset( "labels", data=split_Y_train[i] )
    outfile.close()
