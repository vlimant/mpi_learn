#!/usr/bin/env python

### Set up MPI master and worker processes

import numpy as np
import argparse
from mpi4py import MPI

from mpi_tools.MPIManager import MPIManager
from mpi_tools.Algo import Algo

parser = argparse.ArgumentParser()
parser.add_argument('--masters', help='number of master processes', default=1, type=int)
parser.add_argument('--steps', help='number of training steps', default=10, type=int)
args = parser.parse_args()

comm = MPI.COMM_WORLD.Dup()

manager = MPIManager( comm=comm, num_masters=args.masters, train_steps=args.steps )

# master defines model and propagates it
if comm.Get_rank() == 0:

    # define model to train
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    model = Sequential()
    model.add(Dense(output_dim=64, input_dim=100))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=10))
    model.add(Activation("softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    # parameters for process
    model_arch = model.to_json()
    algo = Algo()
    weights = model.get_weights()

    manager.process.set_model_info( model_arch, algo, weights )

    # initiate training
    manager.process.train()

    # clean up
    manager.free_comms()
