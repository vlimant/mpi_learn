#!/usr/bin/env python

### Set up MPI master and worker processes

import numpy as np
import argparse
from mpi4py import MPI

from mpi_tools.MPIManager import MPIManager
from mpi_tools.Algo import Algo
import Model

parser = argparse.ArgumentParser()
parser.add_argument('--masters', help='number of master processes', default=1, type=int)
parser.add_argument('--steps', help='number of training steps', default=10, type=int)
parser.add_argument('--load-model', action='store_true', dest='load_model', help='load model from h5')
parser.add_argument('--model-name', dest='model_name', help='Name of predefined model to construct')
args = parser.parse_args()
model_name = args.model_name

comm = MPI.COMM_WORLD.Dup()

manager = MPIManager( comm=comm, num_masters=args.masters, train_steps=args.steps )

# master defines model and propagates it to workers
if comm.Get_rank() == 0:

    # get model
    if args.load_model:
        from keras.models import model_from_json
        json_filename = "%s_arch.json" % model_name
        with open( json_filename ) as arch_file:
            model = model_from_json( arch_file.readline() )
        weights_filename = "%s_weights.h5" % model_name
        model.load_weights( weights_filename )
    elif model_name is not None:
        model = Model.make_model( model_name )

    # get model parameters
    model_arch = model.to_json()
    algo = Algo()
    weights = model.get_weights()

    manager.process.set_model_info( model_arch, algo, weights )

    # initiate training
    manager.process.train() 
    # clean up
    manager.free_comms()
