#!/usr/bin/env python

### This script creates a Keras model and a Manager object that handles distributed training.

import sys,os
import numpy as np
import argparse
from mpi4py import MPI
from time import sleep

from mpi_tools.MPIManager import MPIManager, get_device
from Algo import AdaDelta
from Data import H5Data

def load_model(model_name, load_weights):
    """Loads model architecture from <model_name>_arch.json.
        If load_weights is True, gets model weights from
        <model_name_weights.h5"""
    json_filename = "%s_arch.json" % model_name
    with open( json_filename ) as arch_file:
        model = model_from_json( arch_file.readline() ) 
    if load_weights:
        weights_filename = "%s_weights.h5" % model_name
        model.load_weights( weights_filename )
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', help=('will load model architecture from '
                                            '<model_name>_arch.json'))
    parser.add_argument('train_data', help='text file listing data inputs for training')
    parser.add_argument('val_data', help='text file listing data inputs for validation')
    parser.add_argument('--load-weights',help='load weights from <model_name>_weights.h5',action='store_true')
    parser.add_argument('--masters', help='number of master processes', default=1, type=int)
    parser.add_argument('--epochs', help='number of training epochs', default=1, type=int)
    parser.add_argument('--batch', help='batch size', default=100, type=int)
    parser.add_argument('--learning-rate', help='learning rate for SGD', default=0.01, type=float)
    parser.add_argument('--max-gpus', dest='max_gpus', help='max GPUs to use', 
            type=int, default=-1)
    parser.add_argument('--validate-every', help='how long to wait between validations', 
            type=int, default=1000, dest='validate_every')
    parser.add_argument('--features-name', help='name of HDF5 dataset with input features',
            default='features', dest='features_name')
    parser.add_argument('--labels-name', help='name of HDF5 dataset with output labels',
            default='labels', dest='labels_name')
    args = parser.parse_args()
    args = parser.parse_args()
    model_name = args.model_name

    with open(args.train_data) as train_list_file:
        train_list = [ s.strip() for s in train_list_file.readlines() ]
    with open(args.val_data) as val_list_file:
        val_list = [ s.strip() for s in val_list_file.readlines() ]

    comm = MPI.COMM_WORLD.Dup()
    # We have to assign GPUs to processes before importing Theano.
    device = get_device( comm, args.masters, gpu_limit=args.max_gpus )
    print "Process",comm.Get_rank(),"using device",device
    os.environ['THEANO_FLAGS'] = "device=%s,floatX=float32" % (device)
    import theano

    # There is an issue when multiple processes import Keras simultaneously --
    # the file .keras/keras.json is sometimes not read correctly.  
    # as a workaround, just try several times to import keras.
    # Note: importing keras imports theano -- 
    # impossible to change GPU choice after this.
    for try_num in range(10):
        try:
            from keras.models import model_from_json
            break
        except ValueError:
            print "Unable to import keras. Trying again: %d" % try_num
            sleep(0.1)

    data = H5Data( None, batch_size=args.batch, val_samples=1000, 
            features_name=args.features_name, labels_name=args.labels_name )
    # Creating the MPIManager object causes all needed worker and master nodes to be created
    manager = MPIManager( comm=comm, data=data, num_epochs=args.epochs, 
            train_list=train_list, val_list=val_list, num_masters=args.masters )

    # Process 0 defines the model and propagates it to the workers.
    if comm.Get_rank() == 0:

        model = load_model(model_name, load_weights=args.load_weights)

        model_arch = model.to_json()
        algo = AdaDelta( loss='categorical_crossentropy', validate_every=args.validate_every )
        weights = model.get_weights()

        manager.process.set_model_info( model_arch, algo, weights )
        manager.process.train() 
        manager.free_comms()
