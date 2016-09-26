### MPIWorker and MPIMaster classes 

import time
import os,sys
import numpy as np
from mpi4py import MPI
from keras.models import model_from_json
from keras.optimizers import SGD

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
           gradient: latest gradient obtained from training
           data: Data object used to generate training or validation data
    """

    def __init__(self, parent_comm, parent_rank=None, data=None):
        """If the rank of the parent is given, initialize this process and immediately start 
            training. If no parent is indicated, model information should be set manually
            with set_model_info() and training should be launched with train().
            
            Parameters:
              parent_comm: MPI intracommunicator used to communicate with parent
              parent_rank (integer): rank of this node's parent in parent_comm
              data: Data object used to generate training or validation data
        """
        self.parent_comm = parent_comm 
        self.parent_rank = parent_rank
        self.rank = parent_comm.Get_rank()
        self.data = data
        self.model = None
        self.model_arch = None
        self.algo = None
        self.weights_shapes = None
        self.weights = None
        self.gradient = None

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
            self.gradient = weights_from_shapes( self.weights_shapes )

    def check_sanity(self):
        """Throws an exception if any model attribute has not been set yet."""
        for par in ['model','model_arch','algo','weights_shapes','weights']:
            if not hasattr(self, par) or getattr(self, par) is None:
                raise Error("%s not found!  Did you call initialize() for this process?" % par)

    def train(self):
        """To be implemented in derived classes"""
        raise NotImplementedError

    def compile_model(self):
        """Compile the model. Workers are only responsible for computing the gradient and 
            sending it to the master, so we use ordinary SGD with learning rate 1 and 
            compute the gradient as (old weights - new weights) after each batch"""
        sgd = SGD(lr=1.0)
        print "Process %d compiling model" % self.rank
        self.model.compile( loss=self.algo.loss, optimizer=sgd, metrics=['accuracy'] )

    def print_metrics(self, metrics):
        """Display metrics computed during training or validation"""
        names = self.model.metrics_names
        if len(names) == 1:
            print "%s: %.3f" % (names[0],metrics)
        else:
            for i,m in enumerate(names):
                print "%s: %.3f" % (m,metrics[i]),
            print ""

    ### MPI-related functions below ###

    # This dict associates message strings with integers to be passed as MPI tags.
    tag_lookup = {
            'any':          MPI.ANY_SOURCE,
            'train':          0,
            'exit':           1,
            'expect_weights': 2,
            'expect_gradient':3,
            'model_arch':     10,
            'algo':           11,
            'weights_shapes': 12,
            'weights':        13,
            'gradient':       14,
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

    def send_arrays(self, obj, expect_tag, tag, comm=None, dest=None):
        """Send a list of numpy arrays to the process specified by comm (MPI communicator) and dest (rank).
            We first send expect_tag to tell the dest process that we are sending several buffer objects,
            then send the objects layer by layer."""
        self.send( None, expect_tag, comm=comm, dest=dest )
        for w in obj:
            self.send( w, tag, comm=comm, dest=dest, buffer=True )

    def send_weights(self, comm=None, dest=None):
        """Send NN weights to the process specified by comm (MPI communicator) and dest (rank).
            Before sending the weights we first send the tag 'expect_weights'."""
        self.send_arrays( self.weights, expect_tag='expect_weights', tag='weights', 
                comm=comm, dest=dest )

    def send_gradient(self, comm=None, dest=None):
        """Send gradient to the process specified by comm (MPI communicator) and dest (rank).
            Before sending the gradient we first send the tag 'expect_gradient'"""
        self.send_arrays( self.gradient, expect_tag='expect_gradient', tag='gradient', 
                comm=comm, dest=dest )

    def recv_arrays(self, obj, tag, comm=None, source=None):
        """Receive a list of numpy arrays from the process specified by comm (MPI communicator) 
            and dest (rank).
              obj: list of destination arrays 
              tag: MPI tag accompanying the message"""
        for w in obj:
            self.recv( w, tag, comm=comm, source=source, buffer=True )

    def recv_weights(self, comm=None, source=None):
        """Receive NN weights layer by layer from the process specified by comm and source"""
        self.recv_arrays( self.weights, tag='weights', comm=comm, source=source )
        self.model.set_weights( self.weights )

    def recv_gradient(self, comm=None, source=None):
        """Receive gradient layer by layer from the process specified by comm and source"""
        self.recv_arrays( self.gradient, tag='gradient', comm=comm, source=source )

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
          num_epochs: integer giving the number of epochs to train for
    """

    def __init__(self, data, parent_comm, parent_rank=None, num_epochs=1):
        """Raises an exception if no parent rank is provided. Sets the number of epochs 
            using the argument provided, then calls the parent constructor"""
        if parent_rank is None:
            raise Error("MPIWorker initialized without parent rank")
        self.num_epochs = num_epochs
        info = "Creating MPIWorker with rank {0} and parent rank {1} on a communicator of size {2}" 
        print info.format(parent_comm.Get_rank(),parent_rank, parent_comm.Get_size())
        super(MPIWorker, self).__init__( parent_comm, parent_rank, data=data )

    def train(self):
        """Compile the model, then wait for the signal to train. Then train for num_epochs epochs.
            In each step, train on one batch of input data, then send the gradient to the master
            and wait to receive a new set of weights.  When done, send 'exit' signal to parent.
        """
        self.check_sanity()
        self.compile_model()
        self.await_signal_from_parent()
        for epoch in range(self.num_epochs):
            print "MPIWorker %d beginning epoch %d" % (self.rank, epoch)
            for batch in self.data.generate_data():
                self.train_on_batch(batch)
                self.compute_gradient()
                self.send_gradient()
                self.recv_weights()
        print "MPIWorker %d signing off" % self.rank
        self.send_exit_to_parent()

    def train_on_batch(self, batch):
        """Train on a single batch"""
        train_loss = self.model.train_on_batch( batch[0], batch[1] )
        print "Training metrics:",
        self.print_metrics(train_loss)

    def compute_gradient(self):
        """Compute the gradient from the new and old sets of model weights"""
        self.gradient = []
        for i in range(len(self.weights_shapes)):
            self.gradient.append( np.subtract( self.weights[i], self.model.get_weights()[i] ) )

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
          num_updates: number of weight updates received from workers
          best_val_loss: best validation loss computed so far during training
    """

    def __init__(self, parent_comm, parent_rank=None, child_comm=None, data=None):
        """Parameters:
              child_comm: MPI communicator used to contact children"""
        if child_comm is None:
            raise Error("MPIMaster initialized without child communicator")
        self.child_comm = child_comm
        self.has_parent = False
        if parent_rank is not None:
            self.has_parent = True
        self.best_val_loss = None
        self.num_workers = child_comm.Get_size() - 1 #all processes but one are workers
        info = ("Creating MPIMaster with rank {0} and parent rank {1}. "
                "(Communicator size {2}, Child communicator size {3})")
        print info.format(parent_comm.Get_rank(),parent_rank,parent_comm.Get_size(), 
                child_comm.Get_size())
        super(MPIMaster, self).__init__( parent_comm, parent_rank, data=data )

    def process_message(self, status):
        """Extracts message source and tag from the MPI status object and processes the message. 
            Returns the tag of the message received.
            Possible messages are:
            -expect_gradient: worker is ready to send a new gradient
            -exit: worker is done training and will shut down
        """
        source = status.Get_source()
        tag = self.lookup_mpi_tag( status.Get_tag(), inv=True )
        if tag == 'expect_gradient':
            self.recv_gradient( source=source, comm=self.child_comm )
            self.apply_update()
            self.num_updates += 1
            self.send_weights( dest=source, comm=self.child_comm )
            if self.has_parent:
                self.send_gradient()
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
        self.compile_model()
        self.signal_children()

        status = MPI.Status()
        self.running_workers = self.num_workers
        
        self.num_updates = 0
        while self.running_workers > 0:
            self.recv_any_from_child(status)
            self.process_message( status )
            if self.num_updates >= self.algo.validate_every:
                self.validate()
        print "MPIMaster %d done training" % self.rank
        self.validate()
        self.send_exit_to_parent()

    def validate(self, save_if_best=True):
        """Reset the updates counter and compute the loss on the validation data.
            If save_if_best is true, save the model if the validation loss is the 
            smallest so far."""
        if self.has_parent:
            return
        self.num_updates = 0
        self.model.set_weights(self.weights)

        n_batches = 0
        val_metrics = [ 0.0 for i in range( len(self.model.metrics) ) ]
        for batch in self.data.generate_data():
            n_batches += 1
            val_metrics = np.add( val_metrics, self.model.test_on_batch(*batch) )
        val_metrics = np.divide( val_metrics, n_batches )
        print "Validation metrics:",
        self.print_metrics(val_metrics)
        if save_if_best:
            self.save_model_if_best(val_metrics)

    def apply_update(self):
        """Updates weights according to gradient received from worker process"""
        self.weights = self.algo.apply_update( self.weights, self.gradient )

    def save_model_if_best(self, val_metrics):
        """If the validation loss is the lowest on record, save the model.
            The output file name is mpi_learn_model.h5"""
        if hasattr( val_metrics, '__getitem__'):
            val_loss = val_metrics[0]
        else:
            val_loss = val_metrics

        if self.best_val_loss is None or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            print "Saving model to mpi_learn_model.h5"
            self.model.save('mpi_learn_model.h5')

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
