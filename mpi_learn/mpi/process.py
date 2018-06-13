### MPIWorker and MPIMaster classes 

import os,sys
import numpy as np
import socket
import time

from mpi4py import MPI

from ..utils import Error, weights_from_shapes, shapes_from_weights
from ..train.model import MPICallbacks
### Classes ###

class MPIProcess(object):
    """Base class for processes that communicate with one another via MPI.  

       Attributes:
           verbose: display verbose output
           parent_comm: MPI intracommunicator used to communicate with this process's parent
           parent_rank (integer): rank of this node's parent in parent_comm
           rank (integer): rank of this node in parent_comm
           model: Keras model to train
           model_builder: ModelBuilder object specifying model
           algo: Algo object defining how to optimize model weights
           weights_shapes: list of tuples indicating the shape of each layer of the model
           weights: list of numpy arrays storing the last weights received from the parent
           update: latest update obtained from training
           data: Data object used to generate training or validation data
           time_step: for keeping track of time
           num_epochs: integer giving the number of epochs to train for
           callbacks_list: list of keras callbacks
           callbacks: keras CallbackList object holding the designated callbacks
           callback_model: model used for callbacks
           stop_training: becomes true when it is time to stop training
    """

    def __init__(self, parent_comm, process_comm, parent_rank=None, num_epochs=1, data=None, algo=None,
            model_builder=None, callbacks=[], verbose=False, custom_objects={}):
        """If the rank of the parent is given, initialize this process and immediately start 
            training. If no parent is indicated, training should be launched with train().
            
            Parameters:
              parent_comm: MPI intracommunicator used to communicate with parent
              parent_rank (integer): rank of this node's parent in parent_comm
              num_epochs: number of training epochs
              data: Data object used to generate training or validation data
              algo: Algo object used to configure the training process
              model_builder: ModelBuilder object specifying model
              callbacks: list of keras callbacks
              verbose: whether to print verbose output
        """
        self.parent_comm = parent_comm
        self.process_comm = process_comm
        self.parent_rank = parent_rank
        self.num_epochs = num_epochs
        self.data = data
        self.algo = algo
        self.model_builder = model_builder
        self.callbacks_list = callbacks
        self.verbose = verbose
        ev = [
            #'send',
            #'receive',
            #'mpi',
            #'build',
            #'bcast'
            #'loss',
            #'metrics'
        ]
        for extra in ['send','receive','mpi','build','bcast','loss','update','metrics']:
            attr = 'tell_%s'%extra
            setattr( self, attr, bool(extra in ev))
            print (attr, getattr(self, attr))
                   
        self.custom_objects = custom_objects

        self.update = None
        self.stop_training = False
        self.time_step = 0

        self._is_shadow = (self.process_comm is not None and self.process_comm.Get_rank()!=0)

        if self.process_comm is not None and self.process_comm.Get_size() > 1:
            import horovod.common as hvd
            hvd.init(comm=self.process_comm)
            # After PyTorch integration whis must be updated
            import horovod.keras as hvdk
            self.algo.worker_optimizer_obj = hvdk.DistributedOptimizer(self.algo.worker_optimizer_obj)
            #self.callbacks_list.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
        
        self.rank = parent_comm.Get_rank() if parent_comm else 0
        self.ranks = "{0}:{1}:{2}".format(
            MPI.COMM_WORLD.Get_rank(),
            self.parent_comm.Get_rank() if self.parent_comm is not None else '-',
            self.process_comm.Get_rank() if self.process_comm is not None else '-')
        self.build_model()
        if (self.parent_rank is not None and self.parent_comm is not None):
            self.bcast_weights( self.parent_comm )
        if (self.parent_rank is not None and self.parent_comm is not None) or (self.process_comm):
            print ("autostart")
            self.train()

    def is_shadow(self, sync = False):
        """signals that the process is a sub-process and should not act normally"""
        if self.process_comm and sync:
            import inspect
            print ("syncing on the process communicator from",inspect.stack()[1][3])
            self.process_comm.Barrier()
        return self._is_shadow

    def build_model(self):
        """Builds the Keras model and updates model-related attributes"""

        tell_me = self.tell_build
        print("building model",socket.gethostname(),self.ranks)
        self.model = self.model_builder.build_model()

        if tell_me: print ("weight pre-compile",socket.gethostname(),self.ranks)
        self.weights = self.model.get_weights()
        self.compile_model()
        if tell_me: print ("getting weights",socket.gethostname(),self.ranks)
        self.weights = self.model.get_weights()
        if tell_me: print ("formatting update")
        self.update = self.model.format_update()
        if tell_me: print ("done with model",socket.gethostname(),self.ranks)
        if tell_me: print ("moving on",socket.gethostname(),self.ranks)

    def check_sanity(self):
        """Throws an exception if any model attribute has not been set yet."""
        for par in ['model',
                    #'weights_shapes',
                    'weights','update']:
            if not hasattr(self, par) or getattr(self, par) is None:
                raise Error("%s not found!  Process %s does not seem to be set up correctly." % (par,self.ranks))

    def init_callbacks(self, for_worker=False):
        """Prepares all keras callbacks to be used in training.
            Automatically attaches a History callback to the end of the callback list.
            If for_worker is True, leaves out callbacks that only make sense 
            with validation enabled."""
        import keras.callbacks as cbks
        remove_for_worker = [cbks.EarlyStopping, cbks.ModelCheckpoint]
        if for_worker:
            for obj in remove_for_worker:
                self.callbacks_list = [ c for c in self.callbacks_list 
                                        if not isinstance(c, obj) ]
                
        self.model.set_history( cbks.History )
        self.callback = MPICallbacks( self.callbacks_list )
        ## do the handshake
        self.callback.handle( self.model )

    def train(self):
        """To be implemented in derived classes"""
        raise NotImplementedError

    def compile_model(self):
        """Compile the model. Note that the compilation settings
            are relevant only for Workers because the Master updates
            its weights using an mpi_learn optimizer."""
        print ("Process {0} compiling model".format(self.ranks))
        self.algo.compile_model( self.model )
        print ("compiled")
        
    def print_metrics(self, metrics):
        """Display metrics computed during training or validation"""
        self.model.print_metrics( metrics )

    def get_logs(self, metrics, val=False):
        """Get dictionary of logs computed during training.
            If val is True, appends 'val' to the beginning of each metric name"""
        print ("Should not get here ever")
        sys.exit(123)
        if val:
            return { 'val_'+name:np.asscalar(metric) for name, metric in 
                    zip( self.model.metrics_names(), metrics ) }
        else:
            return { name:np.asscalar(metric) for name, metric in 
                    zip( self.model.metrics_names(), metrics ) }

    def do_send_sequence(self):
        """Actions to take when sending an update to parent:
            -Send the update (if the parent accepts it)
            -Sync time and model weights with parent"""
        if self.is_shadow():
            return            
            #pass
        tell = self.tell_send
        if tell: print (self.ranks,"start send sequence",self.time_step)
        self.send_update(check_permission=True)
        if tell: print ("update send")
        self.time_step = self.recv_time_step()
        if tell: print ("new time step", self.time_step)
        self.recv_weights()
        if tell: print ("weights received")
        self.algo.set_worker_model_weights( self.model, self.weights )
        if tell: print ("end send sequence")
        
    ### MPI-related functions below ###

    # This dict associates message strings with integers to be passed as MPI tags.
    tag_lookup = {
            'any':          MPI.ANY_TAG,
            'train':          0,
            'exit':           1,
            'begin_weights':  2,
            'begin_update'  : 3,
            'time':           4,
            'bool':           5,
            'history':        6,
            'weights':        12,
            'update':         12,
            }
    # This dict is for reverse tag lookups.
    inv_tag_lookup = { value:key for key,value in tag_lookup.items() }

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
            print ("Error: not found in tag dictionary: {0} -- returning None".format(name))
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
        if tag == 'history':
            obj = comm.recv( source=source, tag=tag_num, status=status )
            return obj
        #if tag in ['bool','time']:
        #    comm.Recv(obj, source=source, tag=tag_num, status=status )
        #    return obj
        if buffer:
            if type(obj) == list:
                for o in obj:
                    if type(o) == list:
                        for sub_o in o:
                            comm.Recv( sub_o, source=source, tag=tag_num, status=status )
                    else:
                        comm.Recv( o, source=source, tag=tag_num, status=status )
            else:
                if self.tell_mpi: print ("self.recv",type(obj))
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
        if tag in ['history']:
            comm.send( obj, dest=dest, tag=tag_num )
            return
        #if tag in ['time']:
        #    comm.Send( obj, dest=dest, tag=tag_num )
        #    return
        if buffer:
            if type(obj) == list:
                for o in obj:
                    if type(o) == list:
                        for sub_o in o:
                            comm.Send( sub_o, dest=dest, tag=tag_num )
                    else:
                        comm.Send( o, dest=dest, tag=tag_num )
            else:
                comm.Send( obj, dest=dest, tag=tag_num )
        else:
            if type(obj) == list:
                for o in obj:
                    if type(o) == list:
                        for sub_o in o:
                            comm.Send( sub_o, dest=dest, tag=tag_num )
                    else:
                        comm.Send( o, dest=dest, tag=tag_num )
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
            if type(obj) == list:
                for o in obj:
                    if type(o) == list:
                        for sub_o in o:
                            if self.tell_bcast: print ("BC",type(sub_o))
                            comm.Bcast( sub_o, root=root )
                    else:
                        comm.Bcast(o, root=root )
            else:
                comm.Bcast( obj, root=root )
        else:
            obj = comm.bcast( obj, root=root )
            return obj

    def send_exit_to_parent(self):
        if self.is_shadow( sync = True): return
        """Send exit tag to parent process, if parent process exists"""
        if self.parent_rank is not None:
            self.send( None, 'exit' )

    def send_history_to_parent(self):
        if self.is_shadow():return        
        """Send keras history or dict of keras histories"""
        if self.parent_rank is not None:
            if hasattr(self, 'histories'):
                self.send( obj=self.histories, tag='history' )
            else:
                self.send( obj=self.model.history(), tag='history' )

    def send_arrays(self, obj, expect_tag, tag, comm=None, dest=None, check_permission=False):
        """Send a list of numpy arrays to the process specified by comm (MPI communicator) 
            and dest (rank).  We first send expect_tag to tell the dest process that we 
            are sending several buffer objects, then send the objects layer by layer.
            Optionally check first to see if the update will be accepted by the master"""
        self.send( None, expect_tag, comm=comm, dest=dest )
        if check_permission:
            # To check permission we send the update's time stamp to the master.
            # Then we wait to receive the decision yes/no.
            self.send_time_step( comm=comm, dest=dest )
            decision = self.recv_bool( comm=comm, source=dest )
            if not decision: return
        tell=self.tell_mpi
        if tell: print ("send_arrays [1]",type(obj))
        for o in obj:
            if tell: print ("send_arrays [2]",type(o))
            if type(o) == list:
                for w in o:
                    self.send( w, tag, comm=comm, dest=dest, buffer=True )
            else:
                self.send( o, tag, comm=comm, dest=dest, buffer=True )

    def send_weights(self, comm=None, dest=None, check_permission=False):
        if self.is_shadow():return        
        """Send NN weights to the process specified by comm (MPI communicator) and dest (rank).
            Before sending the weights we first send the tag 'begin_weights'."""
        if self.tell_send:
            print (self.ranks,"sending",np.ravel(self.weights[0][0])[:10])
            print (self.ranks,"sending",np.ravel(self.weights[1][0])[:10])
        self.send_arrays( self.weights, expect_tag='begin_weights', tag='weights', 
                comm=comm, dest=dest, check_permission=check_permission )

    def send_update(self, comm=None, dest=None, check_permission=False):
        if self.is_shadow():return        
        """Send update to the process specified by comm (MPI communicator) and dest (rank).
            Before sending the update we first send the tag 'begin_update'"""
        if self.tell_send:
            print (self.ranks,"sending update",np.ravel(self.update[0][0])[:10])
            print (self.ranks,"sending update",np.ravel(self.update[1][0])[:10])
        self.send_arrays( self.update, expect_tag='begin_update', tag='update', 
                comm=comm, dest=dest, check_permission=check_permission )

    def send_time_step(self, comm=None, dest=None):
        if self.is_shadow():return        
        """Send the current time step"""
        #print ("sending the time step", self.time_step, type(self.time_step))
        self.send( obj=self.time_step, tag='time', dest=dest, comm=comm )

    def send_bool(self, obj, comm=None, dest=None):
        if self.is_shadow():return        
        self.send( obj=obj, tag='bool', dest=dest, comm=comm )

    def recv_arrays(self, obj, tag, comm=None, source=None, add_to_existing=False):
        """Receive a list of numpy arrays from the process specified by comm (MPI communicator) 
            and dest (rank).
              obj: list of destination arrays 
              tag: MPI tag accompanying the message
              add_to_existing: if true, add to existing object instead of replacing"""
        if add_to_existing:
            print ("this needs some work")
            tmp = weights_from_shapes( [ w.shape for w in obj ] )
            self.recv_arrays( tmp, tag=tag, comm=comm, source=source )
            for i in range(len(obj)):
                obj[i] += tmp[i]
            return
        tell=self.tell_mpi        
        if tell: print ("recv_arrays [1]",type(obj))
        for o in obj:
            if tell: print ("recv_arrays [2]",type(o))
            if type(o) == list:
                for w in o:
                    self.recv( w, tag, comm=comm, source=source, buffer=True )
            else:
                self.recv( o, tag, comm=comm, source=source, buffer=True )

    def recv_weights(self, comm=None, source=None, add_to_existing=False):
        """Receive NN weights layer by layer from the process specified by comm and source"""
        if self.is_shadow():return        
        if self.tell_receive:
            print (self.ranks,"before receiving",np.ravel(self.weights[0][0])[:10])
            print (self.ranks,"before receiving",np.ravel(self.weights[1][0])[:10])
        self.recv_arrays( self.weights, tag='weights', comm=comm, source=source,
                add_to_existing=add_to_existing )
        if self.tell_receive:
            print (self.ranks,"after receiving",np.ravel(self.weights[0][0])[:10])
            print (self.ranks,"after receiving",np.ravel(self.weights[1][0])[:10])

    def recv_update(self, comm=None, source=None, add_to_existing=False):
        """Receive an update layer by layer from the process specified by comm and source.
            Add it to the current update if add_to_existing is True, 
            otherwise overwrite the current update"""
        if self.is_shadow():return        
        self.recv_arrays( self.update, tag='update', comm=comm, source=source,
                add_to_existing=add_to_existing )

    def recv_time_step(self, comm=None, source=None):
        """Receive the current time step"""
        if self.is_shadow():return
        return self.recv( tag='time', comm=comm, source=source )

    def recv_bool(self, comm=None, source=None):
        if self.is_shadow():return        
        return self.recv( tag='bool', comm=comm, source=source )

    def recv_exit_from_parent(self):
        ir = None
        if not self.is_shadow():
            ir = self.parent_comm.irecv( source=0, tag=self.lookup_mpi_tag('exit') )
        elif self.process_comm:
            ir = self.process_comm.irecv( source=0, tag=self.lookup_mpi_tag('exit') )
        ## should bcast to all ranks in process_comm
        return ir


    def bcast_weights(self, comm, root=0):
        """Broadcast weights shape and weights (layer by layer) 
            on communicator comm from the indicated root rank"""
        if self.tell_bcast: print(self.ranks,"bcasting weights")
        self.bcast( self.weights, comm=comm, root=root, buffer=True )


                
class MPIWorker(MPIProcess):
    """This class trains its NN model and exchanges weight updates with its parent."""

    def __init__(self, data, algo, model_builder, process_comm, parent_comm, parent_rank=None, 
            num_epochs=1, callbacks=[], verbose=False, custom_objects={}):
        """Raises an exception if no parent rank is provided. Sets the number of epochs 
            using the argument provided, then calls the parent constructor"""
        if parent_rank is None:
            raise Error("MPIWorker initialized without parent rank")
        info = "Creating MPIWorker with rank {0} and parent rank {1} on a communicator of size {2}"
        tell_comm = parent_comm if parent_comm is not None else process_comm
        print (info.format(tell_comm.Get_rank(),
                           parent_rank,
                           tell_comm.Get_size()))

        super(MPIWorker, self).__init__( parent_comm, process_comm, parent_rank,
                num_epochs=num_epochs, data=data, algo=algo, model_builder=model_builder,
                callbacks=callbacks, verbose=verbose, custom_objects=custom_objects )

    def train(self):
        """Wait for the signal to train. Then train for num_epochs epochs.
            In each step, train on one batch of input data, then send the update to the master
            and wait to receive a new set of weights.  When done, send 'exit' signal to parent.
        """
        self.check_sanity()
        self.init_callbacks(for_worker=True)
        #self.callbacks.on_train_begin()
        self.callback.on_train_begin()
        self.await_signal_from_parent()

        # periodically check this request to see if the parent has told us to stop training
        exit_request = self.recv_exit_from_parent()
        for epoch in range(self.num_epochs):
            print ("MPIWorker {0} beginning epoch {1:d}".format(self.ranks, epoch))
            self.callback.on_epoch_begin(epoch)
            epoch_metrics = np.zeros((1,))
            i_batch = 0
            for i_batch, batch in enumerate(self.data.generate_data()):
                self.callback.on_batch_begin(i_batch)
                if self.process_comm:
                    ## broadcast the weights to all processes
                    ## alternative is to load the broadcast callback from horovod and call it here
                    self.bcast_weights( comm=self.process_comm )
                    if self.process_comm.Get_rank()!=0:
                        self.model.set_weights(self.weights)
                train_metrics = self.train_on_batch(batch)
                batch_logs = self.model.get_logs(train_metrics)
                if epoch_metrics.shape != train_metrics.shape:
                    epoch_metrics = np.zeros( train_metrics.shape)
                epoch_metrics += train_metrics
                if self.algo.should_sync():
                    self.compute_update()
                    self.do_send_sequence()
                self.callback.on_batch_end(i_batch, batch_logs)
                if exit_request and exit_request.Test():
                    self.stop_training = True
                    if self.process_comm:
                        for r in range(1, self.process_comm.Get_size()):
                            ## propagate the exit signal to processes of this worker
                            self.send_exit_to_child(r, comm=self.process_comm)
                    print ("MPIWorker {0} received exit request from master".format(self.ranks))
                    break
            if self.stop_training:
                break
            ## broken
            epoch_metrics = epoch_metrics * (1.0/ (i_batch+1))
            if self.tell_metrics:
                print ("Worker {0} average metrics:".format(self.ranks))
                self.model.print_metrics(epoch_metrics)
            self.callback.on_epoch_end(epoch, metrics = epoch_metrics)
        print ("MPIWorker {0} signing off".format(self.ranks))
        self.send_exit_to_parent()
        self.callback.on_train_end()
        self.send_history_to_parent()
        self.data.finalize()

    def train_on_batch(self, batch):
        """Train on a single batch"""
        train_loss = self.model.train_on_batch( x=batch[0], y=batch[1] )
        if self.tell_metrics:
            print ("Worker {0} metrics:".format(self.ranks))
            self.print_metrics(train_loss)
        if self.tell_loss:
            print (self.ranks,"loss",train_loss) 
        return train_loss

    def compute_update(self):
        """Compute the update from the new and old sets of model weights"""
        self.update = self.algo.compute_update( self.weights, self.model.get_weights() )

    def await_signal_from_parent(self):
        """Wait for 'train' signal from parent process"""
        if not self.is_shadow():
            tag = self.recv( tag='train' )
        if self.process_comm:
            if self.process_comm.Get_rank() == 0:
                for r in range(1,self.process_comm.Get_size()):
                    print ("relaying the train tag to",r)
                    self.send( None, tag='train', comm=self.process_comm, dest=r)
            else:
                print ("receiving the train tag")
                self.recv( tag='train', comm=self.process_comm )



class MPIMaster(MPIProcess):
    """This class sends model information to its worker processes and updates its model weights
        according to updates or weights received from the workers.
        
        Attributes:
          child_comm: MPI intracommunicator used to communicate with child processes
          has_parent: boolean indicating if this process has a parent process
          num_workers: integer giving the number of workers that work for this master
          best_val_loss: best validation loss computed so far during training
          running_workers: list of workers not yet done training
          waiting_workers_list: list of workers that sent updates and are now waiting
          num_sync_workers: number of worker updates to receive before performing an update
          update_tag: MPI tag to expect when workers send updates
          histories: model histories collected from workers
          epoch: current epoch number
    """

    def __init__(self, parent_comm, parent_rank=None, child_comm=None, 
            num_epochs=1, data=None, algo=None, model_builder=None, 
            num_sync_workers=1, callbacks=[], verbose=False, custom_objects={}):
        """Parameters:
              child_comm: MPI communicator used to contact children"""
        if child_comm is None:
            raise Error("MPIMaster initialized without child communicator")
        self.child_comm = child_comm
        self.has_parent = False
        if parent_rank is not None:
            self.has_parent = True
        self.best_val_loss = None
        self.histories = {}
        self.num_workers = child_comm.Get_size() - 1 #all processes but one are workers
        self.num_sync_workers = num_sync_workers
        info = ("Creating MPIMaster with rank {0} and parent rank {1}. "
                "(Communicator size {2}, Child communicator size {3})")
        print (info.format(parent_comm.Get_rank(),parent_rank,parent_comm.Get_size(), 
                child_comm.Get_size()))
        if self.num_sync_workers > 1:
            print ("Will wait for updates from {0:d} workers before synchronizing".format(self.num_sync_workers))
        super(MPIMaster, self).__init__( parent_comm, parent_rank, data=data, 
                algo=algo, model_builder=model_builder, num_epochs=num_epochs, 
                callbacks=callbacks, verbose=verbose, custom_objects=custom_objects )

    def decide_whether_to_sync(self):
        """Check whether enough workers have sent updates"""
        return ( len(self.waiting_workers_list) >= self.num_sync_workers )

    def is_synchronous(self):
        return self.num_sync_workers > 1

    def accept_update(self):
        """Returns true if the master should accept the latest worker's update, false otherwise"""
        return (not self.is_synchronous()) or self.algo.staleness == 0
        
    def sync_children(self):
        """Update model weights and signal all waiting workers to work again.
            Send our update to our parent, if we have one"""
        while self.waiting_workers_list:
            child = self.waiting_workers_list.pop()
            self.sync_child(child)

    def sync_child(self, child):
        self.send_time_step( dest=child, comm=self.child_comm )
        self.send_weights( dest=child, comm=self.child_comm )

    def sync_parent(self):
        if self.has_parent:
            self.do_send_sequence()
        else:
            self.time_step += 1 

    def do_update_sequence(self, source):
        """Update procedure:
         -Compute the staleness of the update and decide whether to accept it.
         -If we accept, we signal the worker and wait to receive the update.
         -After receiving the update, we determine whether to sync with the workers.
         -Finally we run validation if we have completed one epoch's worth of updates."""
        #print ("receiving a time step")
        child_time = self.recv_time_step( source=source, comm=self.child_comm )
        #print ("child time",child_time)
        self.algo.staleness = self.time_step - child_time
        accepted = self.accept_update()
        #print ("sending",accepted,"as accepted flag")
        self.send_bool( accepted, dest=source, comm=self.child_comm )
        if accepted:
            self.recv_update( source=source, comm=self.child_comm, 
                    add_to_existing=self.is_synchronous() )
            self.waiting_workers_list.append(source)
            if self.decide_whether_to_sync():
                if self.algo.send_before_apply:
                    self.sync_parent()
                    self.sync_children()
                    self.apply_update()
                else:
                    self.apply_update()
                    self.sync_parent()
                    self.sync_children()
                self.update = self.model.format_update()
            if (self.algo.validate_every > 0 and 
                    self.time_step % self.algo.validate_every == 0 and self.time_step > 0):
                epoch_logs = self.validate()
                #self.callbacks.on_epoch_end(self.epoch, epoch_logs)
                self.callback.on_epoch_end(self.epoch, logs = epoch_logs)
                self.epoch += 1
                #self.callbacks.on_epoch_begin(self.epoch)
                self.callback.on_epoch_begin(self.epoch)
        else:
            self.sync_child(source)

    def do_worker_finish_sequence(self, worker_id):
        """Actions to take when a worker finishes training and returns its history"""
        key = "%d_%d" % (self.rank, worker_id)
        self.histories[key] = self.recv_history_from_child(worker_id)
        self.running_workers.remove(worker_id)
        self.num_sync_workers -= 1

    def process_message(self, status):
        """Extracts message source and tag from the MPI status object and processes the message. 
            Returns the tag of the message received.
            Possible messages are:
            -begin_update: worker is ready to send a new update
            -exit: worker is done training and will shut down
        """
        source = status.Get_source()
        tag = self.lookup_mpi_tag( status.Get_tag(), inv=True )
        if tag == 'begin_update':
            self.do_update_sequence(source)
        elif tag == 'exit':
            self.do_worker_finish_sequence(source)
        else:
            raise ValueError("Tag %s not recognized" % tag)
        return tag

    def shut_down_workers(self):
        """Signal all running workers to shut down"""
        for worker_id in self.running_workers:
            print ("Signaling worker {0:d} to shut down".format(worker_id))
            self.send_exit_to_child( worker_id )

    def train(self):
        """Broadcasts model information to children and signals them to start training.
            Receive messages from workers and processes each message until training is done.
            When finished, signal the parent process that training is complete.
        """
        self.check_sanity()
        self.bcast_weights( comm=self.child_comm )
        self.init_callbacks(for_worker=self.has_parent)
        #self.callbacks.on_train_begin()
        self.callback.on_train_begin()
        self.signal_children()

        status = MPI.Status()
        self.running_workers = list(range(1, self.num_workers+1))
        self.waiting_workers_list = []
        
        self.epoch = 0
        #self.callbacks.on_epoch_begin(self.epoch)
        self.callback.on_epoch_begin(self.epoch)
        while self.running_workers:
            #print ("running workers",sorted(self.running_workers))
            self.recv_any_from_child(status)
            self.process_message( status )
            if (not self.stop_training) and self.callback.stop_training():#_model.stop_training:
                self.shut_down_workers()
                self.stop_training = True
        print ("MPIMaster {0} done training".format(self.ranks))
        # If we did not finish the last epoch, validate one more time.
        # (this happens if the batch size does not divide the dataset size)
        if self.epoch < self.num_epochs:
            epoch_logs = self.validate()
            #self.callbacks.on_epoch_end(self.epoch, epoch_logs)
            self.callback.on_epoch_end(self.epoch, epoch_logs)
        self.histories[str(self.rank)] = self.model.history()
        self.send_exit_to_parent()
        #self.callbacks.on_train_end()
        self.callback.on_train_end()
        self.send_history_to_parent()
        self.algo.save()
        if not self.has_parent:
            return self.histories

    def validate(self):
        """Compute the loss on the validation data.
            Return a dictionary of validation metrics."""
        tell = True
        if self.has_parent:
            return {}
        self.model.set_weights(self.weights)
        if tell: print ("Starting validation")
        #val_metrics = [ 0.0 for i in range( len(self.model.metrics_names()) ) ]
        val_metrics = np.zeros((1,))
        i_batch = 0
        for i_batch, batch in enumerate(self.data.generate_data()):
            #new_val_metrics = self.model.test_on_batch(*batch)
            new_val_metrics = self.model.test_on_batch(x=batch[0], y =batch[1] )
            if val_metrics.shape != new_val_metrics.shape:
                val_metrics =  np.zeros(new_val_metrics.shape)
            val_metrics += new_val_metrics
            #for i in range(len(val_metrics)):
            #    val_metrics[i] += new_val_metrics[i]
        #val_metrics = [ m * 1.0 / (i_batch+1) for m in val_metrics ]
        val_metrics = val_metrics * (1.0/(i_batch+1))
        print ("Validation metrics:")
        self.print_metrics(val_metrics)
        #return self.get_logs(val_metrics, val=True)
        l = self.callback.get_logs(val_metrics, val=True)
        if tell: print ("Ending validation")
        self.data.finalize()
        return l 
        # return self.callback.get_logs(val_metrics, val=True)

    def apply_update(self):
        """Updates weights according to update received from worker process"""
        if self.tell_update:
            print ("rank",self.ranks,"applying weights")
            print (self.ranks,"weights",np.ravel(self.weights[0][0])[:10])
            print (self.ranks,"weights",np.ravel(self.weights[1][0])[:10])
            print (self.ranks,"update",np.ravel(self.update[0][0])[:10])
            print (self.ranks,"update",np.ravel(self.update[1][0])[:10])
        with np.errstate( divide='raise', 
                          invalid='raise', 
                          over='raise',
                          #under ='raise'## might not be too bad
                      ):
            self.weights = self.algo.apply_update( self.weights, self.update )
        if self.tell_update:
            print (self.ranks,"new weights",np.ravel(self.weights[0][0])[:10])
            print (self.ranks,"new weights",np.ravel(self.weights[1][0])[:10])
        
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

    def recv_history_from_child(self, child):
        return self.recv( tag='history', source=child, comm=self.child_comm )

    def send_exit_to_child(self, child, comm=None):
        if comm is None:
            comm = self.child_comm
        return comm.isend( None, dest=child, tag=self.lookup_mpi_tag('exit') )
    
