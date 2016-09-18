### MPIWorker and MPIMaster classes 

import time
import os,sys
import numpy as np
from mpi4py import MPI
import theano.sandbox.cuda
from keras.models import model_from_json

from mpi_tools.Utils import Error, weights_from_shapes, shapes_from_weights

### Classes ###

class MPIProcess(object):
    """Base class for processes that communicate with one another via MPI.  

       Attributes:
           parent_comm: MPI intracommunicator used to communicate with this process's parent
           parent_rank (integer): rank of this node's parent in parent_comm
           rank (integer): rank of this node in parent_comm
           model: Keras model to train
           model_arch: json string giving model architecture information
           algo: Algo object defining how to optimize model weights
           weights_shapes: list of tuples indicating the shape of each layer of the model
           weights: list of numpy arrays storing the last weights received from the parent
    """

    def __init__(self, parent_comm, parent_rank=None, gpu=None):
        """If the rank of the parent is given, initialize this process and immediately start 
            training. If no parent is indicated, model information should be set manually
            with set_model_info() and training should be launched with train().
            
            Parameters:
              parent_comm: MPI intracommunicator used to communicate with parent
              parent_rank (integer): rank of this node's parent in parent_comm
              gpu: integer indicating which GPU should be used with Theano 
                    (None if cpu should be used)
        """
        self.parent_comm = parent_comm 
        self.parent_rank = parent_rank
        self.rank = parent_comm.Get_rank()
        self.model = None
        self.model_arch = None
        self.algo = None
        self.weights_shapes = None
        self.weights = None

        self.set_gpu(gpu)

        if self.parent_rank is not None:
            self.initialize()
            self.train()
        else:
            warning = ("MPIProcess {0} created with no parent rank. "
                        "Please initialize manually")
            print warning.format(self.rank)

    def initialize(self):
        """Receive model, weights, and training algorithm from parent, and store them"""
        self.bcast_model_info( self.parent_comm )
        self.set_model_info( model_arch=self.model_arch, weights=self.weights )

    def set_gpu(self,gpu):
        """Configures theano to use the indicated GPU. 
            gpu: integer indicating which GPU to use"""
        if gpu is None:
            device = 'cpu'
        else:
            device = 'gpu%d' % gpu
        #device = 'gpu0' ###TODO: this is temporary, remove it
        theano.sandbox.cuda.use(device)

    def set_model_info(self, model_arch=None, algo=None, weights=None):
        """Sets NN architecture, training algorithm, and weights.
            Any parameter not provided is skipped."""
        if model_arch is not None:
            self.model_arch = model_arch
            self.model = model_from_json( self.model_arch )
        if algo is not None:
            self.algo = algo
        if weights is not None:
            self.weights = weights
            self.weights_shapes = shapes_from_weights( self.weights )
            self.model.set_weights(self.weights)

    def check_sanity(self):
        """Throws an exception if any model attribute has not been set yet."""
        for par in ['model','model_arch','algo','weights_shapes','weights']:
            if not hasattr(self, par) or getattr(self, par) is None:
                raise Error("%s not found!  Did you call initialize() for this process?" % par)

    def train(self):
        """To be implemented in derived classes"""
        raise NotImplementedError

    ### MPI-related functions below ###

    # This dict associates message strings with integers to be passed as MPI tags.
    tag_lookup = {
            'any':          MPI.ANY_SOURCE,
            'train':         0,
            'exit':          1,
            'expect_weights':2,
            'model_arch':    10,
            'algo':          11,
            'weights_shapes':12,
            'weights':       13,
            }
    # This dict is for reverse tag lookups.
    inv_tag_lookup = { value:key for key,value in tag_lookup.iteritems() }

    def lookup_mpi_tag( self, name, inv=False ):
        """Searches for the indicated name in the tag lookup table and returns it if found.
            Params:
              name: item to be looked up
              inv: boolean that is True if an inverse lookup should be performed (int --> string)"""
        if inv:
            lookup = self.inv_tag_lookup
        else:
            lookup = self.tag_lookup
        try:
            return lookup[name]
        except KeyError:
            print "Error: not found in tag dictionary: %s -- returning None" % name
            return None

    def recv(self, obj=None, tag=MPI.ANY_TAG, source=None, buffer=False, status=None, comm=None):
        """Wrapper around MPI.recv/Recv. Returns the received object.
            Params:
              obj: variable into which the received object should be placed
              tag: string indicating which MPI tag should be received
              source: integer rank of the message source.  Defaults to self.parent_rank
              buffer: True if the received object should be sent as a single-segment buffer
                (e.g. for numpy arrays) using MPI.Recv rather than MPI.recv
              status: MPI status object that is filled with received status information
              comm: MPI communicator to use.  Defaults to self.parent_comm"""
        if comm is None:
            comm = self.parent_comm
        if source is None:
            if self.parent_rank is None:
                raise Error("Attempting to receive %s from parent, but parent rank is None" % tag)
            source = self.parent_rank 
        tag_num = self.lookup_mpi_tag(tag)
        if buffer:
            comm.Recv( obj, source=source, tag=tag_num, status=status )
            return obj
        else:
            obj = comm.recv( source=source, tag=tag_num, status=status )
            return obj

    def send(self, obj, tag, dest=None, buffer=False, comm=None):
        """Wrapper around MPI.send/Send.  Params:
             obj: object to send
             tag: string indicating which MPI tag to send
             dest: integer rank of the message destination.  Defaults to self.parent_rank
             buffer: True if the object should be sent as a single-segment buffer
                (e.g. for numpy arrays) using MPI.Send rather than MPI.send
             comm: MPI communicator to use.  Defaults to self.parent_comm"""
        if comm is None:
            comm = self.parent_comm
        if dest is None:
            if self.parent_rank is None:
                raise Error("Attempting to send %s to parent, but parent rank is None" % tag)
            dest = self.parent_rank
        tag_num = self.lookup_mpi_tag(tag)
        if buffer:
            comm.Send( obj, dest=dest, tag=tag_num )
        else:
            comm.send( obj, dest=dest, tag=tag_num )

    def bcast(self, obj, root=0, buffer=False, comm=None):
        """Wrapper around MPI.bcast/Bcast.  Returns the broadcasted object.
            Params: 
              obj: object to broadcast
              root: rank of the node to broadcast from
              buffer: True if the object should be sent as a single-segment buffer
                (e.g. for numpy arrays) using MPI.Bcast rather than MPI.bcast
              comm: MPI communicator to use.  Defaults to self.parent_comm"""
        if comm is None:
            comm = self.parent_comm
        if buffer:
            comm.Bcast( obj, root=root )
        else:
            obj = comm.bcast( obj, root=root )
            return obj

    def send_exit_to_parent(self):
        """Send exit tag to parent process, if parent process exists"""
        if self.parent_rank is not None:
            self.send( None, 'exit' )

    def send_weights(self, comm=None, dest=None):
        """Send NN weights to the process specified by comm (MPI communicator) and dest (rank).
            We first send the 'expect_weights' tag to signal that we are sending weights,
            and then send the weights layer by layer."""
        self.send( None, 'expect_weights', comm=comm, dest=dest )
        for w in self.weights:
            self.send( w, 'weights', comm=comm, dest=dest, buffer=True )

    def recv_weights(self, comm=None, source=None):
        """Receive NN weights layer by layer from the process specified by comm and dest."""
        for w in self.weights:
            self.recv( w, 'weights', comm=comm, source=source, buffer=True )

    def bcast_weights(self, comm, root=0):
        """Broadcast weights layer by layer on communicator comm from the indicated root rank"""
        for w in self.weights:
            self.bcast( w, comm=comm, root=root, buffer=True )

    def bcast_model_info(self, comm, root=0):
        """Broadcast model architecture, optimization algorithm, and weights shape
            using communicator comm and the indicated root rank"""
        for tag in ['model_arch','algo','weights_shapes']:
            setattr( self, tag, self.bcast( getattr(self, tag), comm=comm, root=root ) )
        if self.weights is None:
            self.weights = weights_from_shapes( self.weights_shapes )
        self.bcast_weights( comm, root )

class MPIWorker(MPIProcess):
    """This class trains its NN model and exchanges weight updates with its parent.

        Attributes:
          train_steps: integer indicating number of training steps to perform
    """

    def __init__(self, parent_comm, parent_rank=None, train_steps=0):
        """Raises an exception if no parent rank is provided. Sets the number of training steps
            using the argument provided, then calls the parent constructor with parent_comm 
            and parent_rank."""
        if parent_rank is None:
            raise Error("MPIWorker initialized without parent rank")
        self.train_steps = train_steps
        info = "Creating MPIWorker with rank {0} and parent rank {1} on a communicator of size {2}" 
        print info.format(parent_comm.Get_rank(),parent_rank, parent_comm.Get_size())
        super(MPIWorker, self).__init__( parent_comm, parent_rank )

    def train(self):
        """After receiving the 'train' signal from the parent, train the model for train_steps steps.
            In each step, train on one batch of input data, then send the gradient to the master
            and weight to receive a new set of weights.  When done, send 'exit' signal to parent.
        """
        self.check_sanity()
        self.await_signal_from_parent()
        print "MPIWorker %d beginning training" % self.rank
        for step in range(self.train_steps):
            #pretend to train
            time.sleep(3)
            self.send_weights()
            self.recv_weights()
        print "MPIWorker %d signing off" % self.rank
        self.send_exit_to_parent()

    def await_signal_from_parent(self):
        """Wait for 'train' signal from parent process"""
        self.recv( tag='train' )

class MPIMaster(MPIProcess):
    """This class sends model information to its worker processes and updates its model weights
        according to gradients received from the workerss.
        
        Attributes:
          child_comm: MPI intracommunicator used to communicate with child processes
          has_parent: boolean indicating if this process has a parent process
          num_workers: integer giving the number of workers that work for this master
    """

    def __init__(self, parent_comm, parent_rank=None, child_comm=None):
        """Parameters:
              child_comm: MPI communicator used to contact children"""
        if child_comm is None:
            raise Error("MPIMaster initialized without child communicator")
        self.child_comm = child_comm
        self.has_parent = False
        if parent_rank is not None:
            self.has_parent = True
        self.num_workers = child_comm.Get_size() - 1 #all processes but one are workers
        info = ("Creating MPIMaster with rank {0} and parent rank. "
                "(Communicator size {1}, Child communicator size {2})")
        print info.format(parent_comm.Get_rank(),parent_rank,parent_comm.Get_size(), 
                child_comm.Get_size())
        super(MPIMaster, self).__init__( parent_comm, parent_rank )

    def process_message(self, status):
        """Extracts message source and tag from the MPI status object and processes the message. 
            Returns the tag of the message received.
            Possible messages are:
            -expect_weights: worker is ready to send a new set of weights
            -exit: worker is done training and will shut down
        """
        source = status.Get_source()
        tag = self.lookup_mpi_tag( status.Get_tag(), inv=True )
        if tag == 'expect_weights':
            self.recv_weights( source=source, comm=self.child_comm )
            self.apply_update( self.new_weights )
            self.send_weights( dest=source, comm=self.child_comm )
            if self.has_parent:
                self.send_weights()
                self.recv_weights()
        elif tag == 'exit':
            self.running_workers -= 1 
        else:
            raise ValueError("Tag %s not recognized" % tag)
        return tag

    def train(self):
        """Broadcasts model information to children and signals them to start training.
            Receive messages from workers and processes each message until training is done.
            When finished, signal the parent process that training is complete.
        """
        self.check_sanity()
        self.bcast_model_info( comm=self.child_comm )
        self.signal_children()

        status = MPI.Status()
        self.new_weights = weights_from_shapes( self.weights_shapes )
        self.running_workers = self.num_workers
        
        while self.running_workers > 0:
            self.recv_any_from_child(status)
            self.process_message( status )
        print "MPIMaster %d done training" % self.rank
        self.send_exit_to_parent()

    def apply_update(self, new_weights):
        '''PLACEHOLDER'''
        pass

    ### MPI-related functions below

    def signal_children(self):
        """Sends each child a message telling them to start training"""
        for child in range(1, self.child_comm.Get_size()):
            self.send( obj=None, tag='train', dest=child, comm=self.child_comm )

    def recv_any_from_child(self,status):
        """Receives any message from any child.  Returns the provided status object,
            populated with information about received message"""
        self.recv( tag='any', source=MPI.ANY_SOURCE, status=status, comm=self.child_comm )
        return status
