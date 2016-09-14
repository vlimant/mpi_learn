### MPIWorker and MPIMaster classes 

import time
import sys
import os.path
import numpy as np
from mpi4py import MPI

from keras.models import model_from_json

### Utilities ###

def weights_from_shapes(weights_shapes):
    return [ np.empty( shape, dtype=np.float32 ) for shape in weights_shapes ]

def shapes_from_weights(weights):
    return [ w.shape for w in weights ]

### Classes ###

class MPIProcess(object):
    '''Base class for processes that communicate with one another via MPI.  
       Constructor arguments:
           parent_comm: MPI intracommunicator
           parent_rank: rank of this node's parent in the given communicator'''

    def __init__(self, parent_comm, parent_rank=None):
        self.parent_comm = parent_comm 
        self.parent_rank = parent_rank
        self.rank = parent_comm.Get_rank()
        self.model = None
        self.model_arch = None
        self.algo = None
        self.weights_shapes = None
        self.weights = None

        #If a parent process is indicated, process will automatically set itself up and await the signal to train
        if self.parent_rank is not None:
            self.initialize()
            self.train()
        else:
            #Driver program needs to call set_model_info() and train() on top-level process
            print "MPIProcess %d created with no parent rank.  Please initialize manually" % self.rank

    def initialize(self):
        '''Receive model, weights, and training algorithm from parent, and store them'''
        self.bcast_model_info( self.parent_comm )
        self.set_model_info( model_arch=self.model_arch, weights=self.weights )

    def set_model_info(self, model_arch=None, algo=None, weights=None):
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
        for par in ['model','model_arch','algo','weights_shapes','weights']:
            if not hasattr(self, par) or getattr(self, par) is None:
                raise RuntimeError("%s not found!  Did you call initialize() for this process?" % par)

    def train(self):
        '''To be implemented in derived classes'''
        raise NotImplementedError

    ### MPI-related functions below ###

    # MPI tag lookup table
    tag_lookup = {
            # signals during training
            'any':          MPI.ANY_SOURCE,
            'train':         0,
            'exit':          1,
            'expect_weights':2,
            # for exchanging model info
            'model_arch':    10,
            'algo':          11,
            'weights_shapes': 12,
            'weights':       13,
            }
    inv_tag_lookup = { value:key for key,value in tag_lookup.iteritems() }

    def lookup_mpi_tag( self, name, inv=False ):
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
        '''All MPI receives should go through here'''
        if comm is None:
            #by default use parent communicator
            comm = self.parent_comm
        if source is None:
            #by default receive from parent
            if self.parent_rank is None:
                raise RuntimeError("Attempting to receive %s from parent, but parent rank is None" % tag)
            source = self.parent_rank 
        #receive either as numpy array (Recv) or as pickle (recv)
        tag_num = self.lookup_mpi_tag(tag)
        if buffer:
            comm.Recv( obj, source=source, tag=tag_num, status=status )
            return obj
        else:
            obj = comm.recv( source=source, tag=tag_num, status=status )
            return obj

    def send(self, obj, tag, dest=None, buffer=False, comm=None):
        '''All MPI sends should go through here'''
        if comm is None:
            #by default use parent communicator
            comm = self.parent_comm
        if dest is None:
            #by default send to parent
            if self.parent_rank is None:
                raise RuntimeError("Attempting to send %s to parent, but parent rank is None" % tag)
            dest = self.parent_rank
        #send either as numpy array (Send) or as pickle (send)
        tag_num = self.lookup_mpi_tag(tag)
        if buffer:
            comm.Send( obj, dest=dest, tag=tag_num )
        else:
            comm.send( obj, dest=dest, tag=tag_num )

    def bcast(self, obj, root=0, buffer=False, comm=None):
        '''All MPI broadcasts should go through here'''
        if comm is None:
            #by default use parent communicator
            comm = self.parent_comm
        #send either as numpy array (Bcast) or as pickle (bcast)
        if buffer:
            comm.Bcast( obj, root=root )
        else:
            obj = comm.bcast( obj, root=root )
            return obj

    def send_exit_to_parent(self):
        if self.parent_rank is not None:
            self.send( None, 'exit' )

    def send_weights(self, comm=None, dest=None):
        #first announce that we are sending weights, then send them
        self.send( None, 'expect_weights', comm=comm, dest=dest )
        for w in self.weights:
            self.send( w, 'weights', comm=comm, dest=dest, buffer=True )

    def recv_weights(self, comm=None, source=None):
        for w in self.weights:
            self.recv( w, 'weights', comm=comm, source=source, buffer=True )

    def bcast_weights(self, comm, root=0):
        for w in self.weights:
            self.bcast( w, comm=comm, root=root, buffer=True )

    def bcast_model_info(self, comm, root=0):
        #objects received as pickle 
        for tag in ['model_arch','algo','weights_shapes']:
            setattr( self, tag, self.bcast( getattr(self, tag), comm=comm, root=root ) )
        #weights received as numpy array
        if self.weights is None:
            self.weights = weights_from_shapes( self.weights_shapes )
        self.bcast_weights( comm, root )

class MPIWorker(MPIProcess):

    def __init__(self, parent_comm, parent_rank=None, train_steps=0):
        if parent_rank is None:
            raise RuntimeError("MPIWorker initialized without parent rank")
        self.train_steps = train_steps
        print "Creating MPIWorker with rank %d and parent rank %d on a communicator of size %d" % (parent_comm.Get_rank(),parent_rank, parent_comm.Get_size())
        super(MPIWorker, self).__init__( parent_comm, parent_rank )

    def train(self):
        self.check_sanity()
        self.await_signal_from_parent()
        print "MPIWorker %d beginning training" % self.rank
        for step in range(self.train_steps):
            #pretend to train
            time.sleep(3)
            self.send_weights()
            self.recv_weights()
        #signal completion
        print "MPIWorker %d signing off" % self.rank
        self.send_exit_to_parent()

    def await_signal_from_parent(self):
        self.recv( tag='train' )

class MPIMaster(MPIProcess):

    def __init__(self, parent_comm, parent_rank=None, child_comm=None):
        if child_comm is None:
            raise RuntimeError("MPIMaster initialized without child communicator")
        self.child_comm = child_comm
        self.has_parent = False
        if parent_rank is not None:
            self.has_parent = True
        self.num_workers = child_comm.Get_size() - 1 #all processes but one are workers
        print "Creating MPIMaster with rank %d and parent rank"%parent_comm.Get_rank(),parent_rank,". (Communicator size %d, Child communicator size %d)" % (parent_comm.Get_size(), child_comm.Get_size())
        super(MPIMaster, self).__init__( parent_comm, parent_rank )

    def process_message(self, status):
        source = status.Get_source()
        tag = self.lookup_mpi_tag( status.Get_tag(), inv=True )
        if tag == 'expect_weights':
            #get new weights (must wait for another message from the worker), update, and send latest weights
            self.recv_weights( source=source, comm=self.child_comm )
            self.apply_update( self.new_weights )
            self.send_weights( dest=source, comm=self.child_comm )
            #also send weights to parent, if any, and receive updated weights
            if self.has_parent:
                self.send_weights()
                self.recv_weights()
        elif tag == 'exit':
            #worker is done training, decrement number of running workers
            self.running_workers -= 1 
        else:
            raise ValueError("Tag %s not recognized" % tag)
        return tag

    def train(self):
        self.check_sanity()
        #initialize children and tell them to train
        self.bcast_model_info( comm=self.child_comm )
        self.signal_children()

        status = MPI.Status()
        self.new_weights = weights_from_shapes( self.weights_shapes )
        self.running_workers = self.num_workers
        
        while self.running_workers > 0:
            #get a message from a worker
            self.recv_any_from_child(status)
            self.process_message( status )
        print "MPIMaster %d done training" % self.rank
        self.send_exit_to_parent()

    def apply_update(self, new_weights):
        '''PLACEHOLDER'''
        pass

    ### MPI-related functions below

    def signal_children(self):
        for child in range(1, self.child_comm.Get_size()):
            self.send( obj=None, tag='train', dest=child, comm=self.child_comm )

    def recv_any_from_child(self,status):
        self.recv( tag='any', source=MPI.ANY_SOURCE, status=status, comm=self.child_comm )
        return status
