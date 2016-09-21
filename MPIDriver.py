#!/usr/bin/env python

### This script creates a Keras model and a Manager object that handles distributed training.

import sys
from os.path import isfile
import numpy as np
import argparse
from mpi4py import MPI
from time import sleep

# There is an issue when multiple processes import Keras simultaneously --
# the file .keras/keras.json is sometimes not read correctly.  
# as a workaround, just try several times to import keras
for try_num in range(10):
    try:
        from keras.models import model_from_json
        break
    except ValueError:
        print "Unable to import keras. Trying again: %d" % try_num
        sleep(0.1)

from mpi_tools.MPIManager import MPIManager
from Algo import VanillaSGD

def load_model(model_name):
    """Loads model architecture from <model_name>_arch.json and gets model weights from
        <model_name_weights.h5"""
    json_filename = "%s_arch.json" % model_name
    with open( json_filename ) as arch_file:
        model = model_from_json( arch_file.readline() ) 
    weights_filename = "%s_weights.h5" % model_name
    model.load_weights( weights_filename )
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', help=('will load model architecture from '
                                            '<model_name>_arch.json and weights from '
                                            '<model_name>_weights.h5'))
    parser.add_argument('train_data', help='text file listing data inputs for training')
    parser.add_argument('val_data', help='text file listing data inputs for validation')
    parser.add_argument('--masters', help='number of master processes', default=1, type=int)
    parser.add_argument('--epochs', help='number of training epochs', default=1, type=int)
    parser.add_argument('--batch', help='batch size', default=100, type=int)
    parser.add_argument('--learning-rate', help='learning rate for SGD', default=0.01, type=float)
    args = parser.parse_args()
    model_name = args.model_name

    with open(args.train_data) as train_list_file:
        train_list = [ s.strip() for s in train_list_file.readlines() ]
    with open(args.val_data) as val_list_file:
        val_list = [ s.strip() for s in val_list_file.readlines() ]

    # Creating the MPIManager object causes all needed worker and master nodes to be created
    comm = MPI.COMM_WORLD.Dup()
    manager = MPIManager( comm=comm, batch_size=args.batch, num_epochs=args.epochs, 
            train_list=train_list, val_list=val_list, num_masters=args.masters )

    # Process 0 defines the model and propagates it to the workers.
    if comm.Get_rank() == 0:

        model = load_model(model_name)

        model_arch = model.to_json()
        algo = VanillaSGD( loss='categorical_crossentropy', learning_rate=args.learning_rate )
        weights = model.get_weights()

        manager.process.set_model_info( model_arch, algo, weights )
        manager.process.train() 
        manager.free_comms()
