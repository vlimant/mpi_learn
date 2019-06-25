#!/usr/bin/env python

### This script creates an MPIManager object and launches distributed training.

import sys,os
import numpy as np
import argparse
import json
import re

from mpi4py import MPI
from time import time,sleep

from mpi_learn.mpi.manager import MPIManager, get_device
from mpi_learn.train.algo import Algo
from mpi_learn.train.data import H5Data
from mpi_learn.train.model import ModelFromJson, ModelTensorFlow
from mpi_learn.utils import import_keras
import socket

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose',help='display metrics for each training batch',action='store_true')
    parser.add_argument('--profile',help='profile theano code',action='store_true')
    parser.add_argument('--monitor',help='Monitor cpu and gpu utilization', action='store_true')
    parser.add_argument('--tf', help='use tensorflow backend', action='store_true')

    # model arguments
    parser.add_argument('model_json', help='JSON file containing model architecture')
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
    parser.add_argument('--preload-data', help='Preload files as we read them', default=0, type=int, dest='data_preload')
    parser.add_argument('--cache-data', help='Cache the input files to a provided directory', default='', dest='caching_dir')

    # configuration of network topology
    parser.add_argument('--masters', help='number of master processes', default=1, type=int)
    parser.add_argument('--processes', help='number of processes per worker', default=1, type=int)
    parser.add_argument('--max-gpus', dest='max_gpus', help='max GPUs to use', 
            type=int, default=-1)
    parser.add_argument('--master-gpu',help='master process should get a gpu',
            action='store_true', dest='master_gpu')
    parser.add_argument('--synchronous',help='run in synchronous mode',action='store_true')

    # configuration of training process
    parser.add_argument('--epochs', help='number of training epochs', default=1, type=int)
    parser.add_argument('--optimizer',help='optimizer for master to use',default='adam')
    parser.add_argument('--loss',help='loss function',default='binary_crossentropy')
    parser.add_argument('--early-stopping', default=None,
                        dest='early_stopping', help='Configuration for early stopping')
    parser.add_argument('--target-metric', default=None,
                        dest='target_metric', help='Passing configuration for a target metric')
    parser.add_argument('--worker-optimizer',help='optimizer for workers to use',
            dest='worker_optimizer', default='sgd')
    parser.add_argument('--worker-optimizer-params',help='worker optimizer parameters (string representation of a dict)',
            dest='worker_optimizer_params', default='{}')
    parser.add_argument('--sync-every', help='how often to sync weights with master', 
            default=1, type=int, dest='sync_every')
    parser.add_argument('--easgd',help='use Elastic Averaging SGD',action='store_true')
    parser.add_argument('--elastic-force',help='beta parameter for EASGD',type=float,default=0.9)
    parser.add_argument('--elastic-lr',help='worker SGD learning rate for EASGD',
            type=float, default=1.0, dest='elastic_lr')
    parser.add_argument('--elastic-momentum',help='worker SGD momentum for EASGD',
            type=float, default=0, dest='elastic_momentum')
    parser.add_argument('--restore', help='pass a file to retore the variables from', default=None)

    args = parser.parse_args()
    model_name = os.path.basename(args.model_json).replace('.json','')

    with open(args.train_data) as train_list_file:
        train_list = [ s.strip() for s in train_list_file.readlines() ]
    with open(args.val_data) as val_list_file:
        val_list = [ s.strip() for s in val_list_file.readlines() ]

    comm = MPI.COMM_WORLD.Dup()

    model_weights = None

    if args.restore:
        args.restore = re.sub(r'\.algo$', '', args.restore)
        if not args.tf:
            model_weights = args.restore + '.model'

    # Theano is the default backend; use tensorflow if --tf is specified.
    # In the theano case it is necessary to specify the device before importing.
    device = get_device( comm, args.masters, gpu_limit=args.max_gpus,
                gpu_for_master=args.master_gpu)
    hide_device = True
    if args.tf: 
        backend = 'tensorflow'
        if not args.optimizer.endswith("tf"):
            args.optimizer = args.optimizer + 'tf'
        if hide_device:
            os.environ['CUDA_VISIBLE_DEVICES'] = device[-1] if 'gpu' in device else ''
            print ('set to device',os.environ['CUDA_VISIBLE_DEVICES'],socket.gethostname())
    else:
        backend = 'theano'
        os.environ['THEANO_FLAGS'] = "profile=%s,device=%s,floatX=float32" % (args.profile,device.replace('gpu','cuda'))
    os.environ['KERAS_BACKEND'] = backend

    print (backend)
    import_keras()
    import keras.backend as K
    if args.tf:
        gpu_options=K.tf.GPUOptions(
            per_process_gpu_memory_fraction=0.1, #was 0.0
            allow_growth = True,
            visible_device_list = device[-1] if 'gpu' in device else '')
        if hide_device:
            gpu_options=K.tf.GPUOptions(
                            per_process_gpu_memory_fraction=0.0,
                            allow_growth = True,)
        K.set_session( K.tf.Session( config=K.tf.ConfigProto(
            allow_soft_placement=True,
            #allow_soft_placement=False,
            #log_device_placement=True , # was false
            log_device_placement=False , # was false
            gpu_options=gpu_options
            ) ) )

    if args.tf:
        #model_builder = ModelTensorFlow( comm, filename=args.model_json, device_name=device , weights=model_weights)
        from mpi_learn.train.GanModel import GANModelBuilder
        model_builder  = GANModelBuilder( comm , device_name=device, tf= True, weights=model_weights)
        print ("Process {0} using device {1}".format(comm.Get_rank(), model_builder.device))
    else:
        from mpi_learn.train.GanModel import GANModelBuilder
        model_builder  = GANModelBuilder( comm , device_name=device, weights=model_weights)
        print ("Process {0} using device {1}".format(comm.Get_rank(),device))
        os.environ['THEANO_FLAGS'] = "profile=%s,device=%s,floatX=float32" % (args.profile,device.replace('gpu','cuda'))
        # GPU ops need to be executed synchronously in order for profiling to make sense
        if args.profile:
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


    data = H5Data( batch_size=args.batch,
                   cache = args.caching_dir,
                   preloading = args.data_preload,
                   features_name=args.features_name, labels_name=args.labels_name )
    # We initialize the Data object with the training data list
    # so that we can use it to count the number of training examples
    data.set_file_names( train_list )
    validate_every = int(data.count_data()/args.batch )

    # Some input arguments may be ignored depending on chosen algorithm
    if args.easgd:
        algo = Algo(None, loss=args.loss, validate_every=validate_every,
                mode='easgd', sync_every=args.sync_every,
                worker_optimizer=args.worker_optimizer,
                worker_optimizer_params=args.worker_optimizer_params,
                elastic_force=args.elastic_force/(min(1,comm.Get_size()-1)),
                elastic_lr=args.elastic_lr, 
                elastic_momentum=args.elastic_momentum) 
    else:
        algo = Algo(args.optimizer, loss=args.loss, validate_every=validate_every,
                sync_every=args.sync_every, worker_optimizer=args.worker_optimizer,
                worker_optimizer_params=args.worker_optimizer_params) 
    if args.restore:
        algo.load(args.restore)

    # Creating the MPIManager object causes all needed worker and master nodes to be created
    manager = MPIManager( comm=comm, data=data, algo=algo, model_builder=model_builder,
                          num_epochs=args.epochs, train_list=train_list, val_list=val_list, 
                          num_masters=args.masters, num_processes=args.processes,
                          synchronous=args.synchronous, 
                          verbose=args.verbose , monitor=args.monitor,
                          early_stopping=args.early_stopping,target_metric=args.target_metric )

    # Process 0 launches the training procedure
    if comm.Get_rank() == 0:
        print (algo)

        t_0 = time()
        histories = manager.process.train() 
        delta_t = time() - t_0
        manager.free_comms()
        print ("Training finished in {0:.3f} seconds".format(delta_t))

        json_name = '_'.join([model_name,args.trial_name,"history.json"]) 
        manager.process.record_details(json_name,
                                       meta={"args":vars(args)})
        print ("Wrote trial information to {0}".format(json_name))
