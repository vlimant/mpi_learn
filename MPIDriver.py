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
from mpi_tools.Algo import Algo
import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--masters', help='number of master processes', default=1, type=int)
    parser.add_argument('--steps', help='number of training steps', default=10, type=int)
    parser.add_argument('--load-model', dest='model_to_load', 
            help=('load model architecture from <model_name>_arch.json and weights from '
                    '<model_name>_weights.h5'))
    parser.add_argument('--build-model', dest='model_to_build', 
            help='Name of predefined model to construct')
    args = parser.parse_args()
    model_to_load = args.model_to_load
    model_to_build = args.model_to_build
    if not (model_to_load is not None or model_to_build is not None):
        sys.exit("Please specify either --load-model <name> or --build-model <name>")

    comm = MPI.COMM_WORLD.Dup()
    manager = MPIManager( comm=comm, num_masters=args.masters, train_steps=args.steps )

    # Process 0 defines the model and propagates it to the workers.
    if comm.Get_rank() == 0:

        # There are two options for creating the model:
        # 1) Load architecture from <model>_arch.json and weights from <model>_weights.h5
        # 2) Build the model from one of the presets in Model.py

        if model_to_load is not None:
            json_filename = "%s_arch.json" % model_to_load
            with open( json_filename ) as arch_file:
                model = model_from_json( arch_file.readline() )
            weights_filename = "%s_weights.h5" % model_to_load
            model.load_weights( weights_filename )
        elif model_to_build is not None:
            model = Model.make_model( model_to_build )
        else:
            print "Model not defined.  Doing nothing.  (execution should not reach here)"
            manager.free_comms()

        model_arch = model.to_json()
        algo = Algo()
        weights = model.get_weights()

        manager.process.set_model_info( model_arch, algo, weights )
        manager.process.train() 
        manager.free_comms()
