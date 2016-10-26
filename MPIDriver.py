#!/usr/bin/env python

### This script creates a Keras model and a Manager object that handles distributed training.

import sys,os
import numpy as np
import argparse
import json
from mpi4py import MPI
from time import time,sleep

from mpi_learn.mpi.manager import MPIManager, get_device
from mpi_learn.train.algo import Algo
from mpi_learn.train.data import H5Data
from mpi_learn.utils import import_keras, load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose',help='display metrics for each training batch',action='store_true')
    parser.add_argument('--profile',help='profile theano code',action='store_true')

    # model arguments
    parser.add_argument('model_name', help=('will load model architecture from '
                                            '<model_name>_arch.json'))
    parser.add_argument('--load-weights',help='load weights from <model_name>_weights.h5',
            action='store_true')
    parser.add_argument('--trial-name', help='descriptive name for trial', 
            default='train', dest='trial_name')

    # training data arguments
    parser.add_argument('train_data', help='text file listing data inputs for training')
    parser.add_argument('val_data', help='text file listing data inputs for validation')
    parser.add_argument('--features-name', help='name of HDF5 dataset with input features',
            default='features', dest='features_name')
    parser.add_argument('--labels-name', help='name of HDF5 dataset with output labels',
            default='labels', dest='labels_name')
    parser.add_argument('--batch', help='batch size', default=100, type=int)

    # configuration of network topology
    parser.add_argument('--masters', help='number of master processes', default=1, type=int)
    parser.add_argument('--max-gpus', dest='max_gpus', help='max GPUs to use', 
            type=int, default=-1)
    parser.add_argument('--master-gpu',help='master process should get a gpu',
            action='store_true', dest='master_gpu')
    parser.add_argument('--synchronous',help='run in synchronous mode',action='store_true')

    # configuration of training process
    parser.add_argument('--epochs', help='number of training epochs', default=1, type=int)
    parser.add_argument('--optimizer',help='optimizer for master to use',default='adam')
    parser.add_argument('--loss',help='loss function',default='binary_crossentropy')
    parser.add_argument('--worker-optimizer',help='optimizer for workers to use',
            dest='worker_optimizer', default='sgd')
    parser.add_argument('--sync-every', help='how often to sync weights with master', 
            default=1, type=int, dest='sync_every')
    parser.add_argument('--easgd',help='use Elastic Averaging SGD',action='store_true')
    parser.add_argument('--elastic-force',help='beta parameter for EASGD',type=float,default=0.9)
    parser.add_argument('--elastic-lr',help='worker SGD learning rate for EASGD',
            type=float, default=1.0, dest='elastic_lr')
    parser.add_argument('--elastic-momentum',help='worker SGD momentum for EASGD',
            type=float, default=0, dest='elastic_momentum')

    args = parser.parse_args()
    model_name = args.model_name

    with open(args.train_data) as train_list_file:
        train_list = [ s.strip() for s in train_list_file.readlines() ]
    with open(args.val_data) as val_list_file:
        val_list = [ s.strip() for s in val_list_file.readlines() ]

    comm = MPI.COMM_WORLD.Dup()
    # We have to assign GPUs to processes before importing Theano.
    device = get_device( comm, args.masters, gpu_limit=args.max_gpus,
            gpu_for_master=args.master_gpu)
    print "Process",comm.Get_rank(),"using device",device
    os.environ['THEANO_FLAGS'] = "profile=%s,device=%s,floatX=float32" % (args.profile,device)
    # GPU ops need to be executed synchronously in order for profiling to make sense
    if args.profile:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    import_keras()
    import keras.callbacks as cbks

    # We initialize the Data object with the training data list
    # so that we can use it to count the number of training examples
    data = H5Data( train_list, batch_size=args.batch, 
            features_name=args.features_name, labels_name=args.labels_name )
    if comm.Get_rank() == 0:
        validate_every = data.count_data()/args.batch 
    callbacks = []
    callbacks.append( cbks.ModelCheckpoint( '_'.join([
        model_name,args.trial_name,"mpi_learn_result.h5"]), 
        monitor='val_loss', verbose=1 ) )

    # Creating the MPIManager object causes all needed worker and master nodes to be created
    manager = MPIManager( comm=comm, data=data, num_epochs=args.epochs, 
            train_list=train_list, val_list=val_list, num_masters=args.masters,
            synchronous=args.synchronous, callbacks=callbacks, verbose=args.verbose )

    # Process 0 defines the model and propagates it to the workers.
    if comm.Get_rank() == 0:
        weights_file = None
        if args.load_weights:
            weights_file = model_name+"_weights.h5"
        model = load_model(model_name+"_arch.json", weights_file=weights_file)
        model_arch = model.to_json()
        if args.easgd:
            algo = Algo(None, loss=args.loss, validate_every=validate_every,
                    mode='easgd', sync_every=args.sync_every,
                    worker_optimizer=args.worker_optimizer,
                    elastic_force=args.elastic_force/(comm.Get_size()-1),
                    elastic_lr=args.elastic_lr, 
                    elastic_momentum=args.elastic_momentum) 
        else:
            algo = Algo(args.optimizer, loss=args.loss, validate_every=validate_every,
                    sync_every=args.sync_every, worker_optimizer=args.worker_optimizer) 
        print algo
        weights = model.get_weights()
        manager.process.set_model_info( model_arch, algo, weights )

        t_0 = time()
        histories = manager.process.train() 
        delta_t = time() - t_0
        manager.free_comms()
        print "Training finished in %.3f seconds" % delta_t

        # Make output dictionary
        out_dict = { "args":vars(args),
                     "history":histories,
                     "train_time":delta_t,
                     }
        json_name = '_'.join([model_name,args.trial_name,"history.json"]) 
        with open( json_name, 'w') as out_file:
            out_file.write( json.dumps(out_dict, indent=4, separators=(',',': ')) )
        print "Wrote trial information to",json_name
