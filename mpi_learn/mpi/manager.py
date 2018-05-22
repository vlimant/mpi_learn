### MPIManager class and associated functions

from __future__ import division
import math
from mpi4py import MPI
import numpy as np

from ..train.data import H5Data
from ..utils import get_num_gpus

def get_master_ranks(comm, num_masters=1):
    """Arguments: 
        comm: MPI intracommunicator containing all processes
        num_masters: number of processes that will be assigned as masters
       Returns:
        a list of integers corresponding to the MPI ranks that will be assigned as masters"""
    if num_masters > 1:
        return [0]+list(range(1, comm.Get_size(), (comm.Get_size()-1) // num_masters))
    return [0]

def get_worker_ranks(comm, num_masters=1):
    """Arguments:
        comm: MPI intracommunicator containing all processes
        num_masters: number of processes that will be assigned as masters
       Returns:
        a list of integers corresponding to the MPI ranks that will be assigned as workers"""
    master_ranks = get_master_ranks( comm, num_masters )
    return [ x for x in range(comm.Get_size()) if x not in master_ranks ]

def get_device(comm, num_masters=1, gpu_limit=-1, gpu_for_master=False):
    """Arguments:
        comm: MPI intracommunicator containing all processes
        num_masters: number of processes that will be assigned as masters
        gpu_limit: maximum number of gpus to use on one host
        gpu_for_master: whether master processes should be given a gpu
       Returns device name 'cpu' or 'gpuN' appropriate for use with theano""" 
    rank = comm.Get_rank()
    if gpu_for_master:
        gpu_ranks = range(comm.Get_size())
    else:
        gpu_ranks = get_worker_ranks( comm, num_masters )

    # Get the ranks of the other processes that share the same host
    # and determine which GPU to take on the host
    host = MPI.Get_processor_name()
    hosts = comm.allgather(host)
    workers_sharing_host = [ i for i in gpu_ranks
            if hosts[i] == host ]
    if rank in workers_sharing_host:
        worker_id = workers_sharing_host.index( rank )
    else:
        worker_id = -1

    print ("gpu ranks",gpu_ranks)
    print ("gpu limit",gpu_limit)

    # get_num_gpus will fail if CUDA is not installed, so we short circuit if 0 GPUs are requested
    if gpu_limit == 0:
        return 'cpu'
    #max_gpu = get_num_gpus() - 1
    #if gpu_limit > 0:
    #    max_gpu = min( max_gpu, gpu_limit-1 )
    #if worker_id < 0:# or worker_id > max_gpu:
    #    return 'cpu'
    #else:
    #    return 'gpu%d' % (worker_id%(max_gpu+1))

    def get_gpu_list(mem_lim = 1):
        import gpustat
        stats = gpustat.GPUStatCollection.new_query()
        ids = list(map(lambda gpu: int(gpu.entry['index']), stats))
        ratios = map(lambda gpu: float(gpu.entry['memory.used'])/float(gpu.entry['memory.total']), stats)
        #used = list(map(lambda gpu: float(gpu.entry['memory.used']), stats))
        #unused_gpu = filter(lambda x: x[1] < 100.0, zip(ids, used))
        print ("GPU usage",[gpu.entry for gpu in stats])
        free = list(map(lambda gpu: float(gpu.entry['memory.total'])-float(gpu.entry['memory.used']), stats))
        unused_gpu = list(filter(lambda x: x[1]  > mem_lim, zip(ids, free)))
        print ("unused",unused_gpu)
        return [x[0] for x in unused_gpu]

    gpu_list = get_gpu_list()
    print ("list of gpu",gpu_list)
    print ("worker id",worker_id)
    if worker_id < 0:
        dev = 'gpu%d' % (gpu_list[0])
        return dev

    if len(gpu_list) == 0:
        print("No free GPU available. Using CPU instead.")
        return 'cpu'
    else:
        max_gpu = len(gpu_list)
        dev = 'gpu%d' % (gpu_list[worker_id % (max_gpu)])
        #print ("Found",dev,"free")
        return dev
                                                                                        
            

class MPIManager(object):
    """The MPIManager class defines the topology of the MPI process network
        and creates master and worker objects for each process accordingly.

        Two configurations are available:
          1) one master supervising other masters, each controlling some workers
          2) one master and N-1 workers, where N is the number of MPI processes

        Attributes:
          process: the MPI worker or master object running on this process
          data: Data object containing information used for training/validation
          algo: Algo object containing training algorithm configuration options
          model_builder: ModelBuilder object
          num_masters: integer indicating the number of master processes.  
            If num_masters > 1, an additional master will be created to supervise all masters.
          num_workers: integer indicating the number of worker processes
          worker_id: ID of worker node, used for indexing training data files
          num_epochs (integer): number of times to iterate over the training data
          comm_block: MPI intracommunicator used for message passing between master and workers.
            Process 0 is the master and the other processes are workers.  
          comm_masters: MPI intracommunicator used for message passing between masters.
            (It will be None if there is only one master.)
          train_list: list of training data file names
          val_list: list of validation data file names
          is_master: boolean determining if this process is a master
          should_validate: boolean determining if this process should run training validation
          synchronous: whether or not to syncronize workers after each update
          callbacks: keras callbacks to use during training
          worker_callbacks: callbacks to be executed by worker processes
          verbose: whether to make MPIProcess objects verbose
    """

    def __init__(self, comm, data, algo, model_builder, num_epochs, train_list, 
            val_list, num_masters=1, synchronous=False, callbacks=[], 
            worker_callbacks=[], verbose=False, custom_objects={}):
        """Create MPI communicator(s) needed for training, and create worker 
            or master object as appropriate.

            Params: 
            comm: MPI intracommunicator containing all processes
            data: Data object containing information used for training/validation
            algo: Algo object containing training algorithm configuration options
            model_builder: ModelBuilder object
            num_masters: number of master processes
            num_epochs: number of times to iterate over the training data
            train_list: list of training data files
            val_list: list of validation data files
            synchronous: true if masters should operate in synchronous mode
            callbacks: list of keras callback objects
            worker_callbacks: list of keras callback objects
            verbose: whether to make MPIProcess objects verbose
        """
        self.data = data
        self.algo = algo
        self.model_builder = model_builder
        self.num_masters = num_masters
        self.num_workers = comm.Get_size() - self.num_masters 
        if self.num_masters > 1:
            self.num_workers -= 1 # one process is taken up by the super-master
        self.worker_id = -1

        self.num_epochs = num_epochs
        self.train_list = train_list
        self.val_list = val_list
        self.synchronous = synchronous
        self.callbacks = callbacks
        self.worker_callbacks = worker_callbacks
        self.verbose = verbose
        self.comm_block = None
        self.comm_masters = None
        self.is_master = None
        self.should_validate = None
        self.custom_objects=custom_objects

        self.make_comms(comm)

    def make_comms(self,comm):
        """Define the network topology by creating communicators linking masters with their slaves.
            Set comm_block to contain one master and all of its workers.
            Set comm_masters to contain all masters, including the "super-master" supervising them.
            Define a master or worker object as appropriate for each process.
            If a worker is created, it is assigned some data files to train on.
        """
        # For masters we let child_comm be the communicator used to message the node's 
        # children, and parent_comm be that used to message the node's parents.

        self.parent_rank = 0

        # Case (1)
        if self.num_masters > 1:
            self.make_comms_many(comm)
            if self.is_master:
                parent_comm = self.comm_masters
                if self.comm_masters.Get_rank() == 0: # rank 0 is the super-master
                    child_comm = self.comm_masters
                    self.parent_rank = None
                else:
                    child_comm = self.comm_block
        # Case (2)
        else:
            self.make_comm_single(comm)
            if self.is_master:
                parent_comm = self.comm_block
                child_comm = self.comm_block
                self.parent_rank = None

        # Process initialization
        from .process import MPIWorker, MPIMaster
        if self.is_master:
            self.set_val_data()
            num_sync_workers = self.get_num_sync_workers(child_comm)
            self.process = MPIMaster( parent_comm, parent_rank=self.parent_rank, 
                    data=self.data, algo=self.algo, model_builder=self.model_builder, 
                    child_comm=child_comm, num_epochs=self.num_epochs, 
                    num_sync_workers=num_sync_workers, callbacks=self.callbacks, 
                    verbose=self.verbose, custom_objects=self.custom_objects)
        else:
            self.set_train_data()
            self.process = MPIWorker( parent_comm=self.comm_block, parent_rank=self.parent_rank, 
                    num_epochs=self.num_epochs, data=self.data, algo=self.algo, 
                    model_builder=self.model_builder, callbacks=self.worker_callbacks, 
                    verbose=self.verbose, custom_objects=self.custom_objects)

    def figure_of_merit(self):
        ##if (self.comm_masters and self.comm_masters.Get_rank() == 0) or (self.comm_block.Get_rank() == 0):
        if self.parent_rank is None:
            ## only the uber-master returns a valid fom
            return self.process.model.figure_of_merit()
        else:
            return None

    def train(self):
        if self.parent_rank is None:
            ## start the uber master, as all over masters are self-started
            ## check MPIProcess.__init__ 
            #if self.parent_rank is not None:
            # self.bcast_weights( self.parent_comm )
            # self.train()
            return self.process.train()
        else:
            return None

    def get_num_sync_workers(self, comm):
        """Returns the number of workers the master should wait for
            at each training time step.  Currently set to 95% of the 
            number of workers (or 1 if running asynchronously).  
            comm should be the master's child communicator."""
        if self.synchronous:
            return int( math.ceil( 0.95 * (comm.Get_size() - 1) ) )
        return 1

    def set_train_data(self):
        """Sets the training data files to be used by the current process"""
        files_per_worker = len(self.train_list) // self.num_workers
        files_for_this_worker = self.train_list[ 
                self.worker_id*files_per_worker : (self.worker_id+1)*files_per_worker ]
        # The worker takes an extra file if needed
        if self.worker_id < len(self.train_list) % self.num_workers:
            files_for_this_worker.append(self.train_list[ self.num_workers*files_per_worker + self.worker_id ])
        print ("Files for worker {0}:".format(self.comm_block.Get_rank()))
        for f in files_for_this_worker:
            print ("  {0}".format(f))
        self.data.set_file_names( files_for_this_worker )

    def set_val_data(self):
        """Sets the validation data files to be used by the current process
            (only the master process has validation data associated with it)"""
        if not self.should_validate: return None
        print ("Files for validation:" )
        for f in self.val_list:
            print ("  {0}".format(f))
        self.data.set_file_names( self.val_list )

    def make_comms_many(self,comm):
        """Create MPI communicators (Case 1):
            Rank 0 of comm_block is the master, other ranks are workers.
            Rank 0 of comm_master is the super-master, other ranks are sub-masters.
            Sets is_master and worker_id attributes."""

        # Create a communicator containing all processes except the first.  
        # Then divide that communicator into blocks, each with one master
        ranks_excludefirstprocess = range(1,comm.Get_size()) 
        comm_excludefirstprocess = comm.Create( comm.Get_group().Incl( ranks_excludefirstprocess ) )
        if comm.Get_rank() in ranks_excludefirstprocess:
            size_block = (comm.Get_size()-1) // self.num_masters
            color_block = comm_excludefirstprocess.Get_rank() // size_block
            self.comm_block = comm_excludefirstprocess.Split( color_block )
            comm_excludefirstprocess.Free()
        else:
            self.comm_block = None
        # Create a communicator containing all masters
        ranks_mastergroup = get_master_ranks( comm, self.num_masters )
        self.comm_masters = comm.Create( comm.Get_group().Incl(ranks_mastergroup) )
        self.is_master = ( comm.Get_rank() in ranks_mastergroup )
        self.should_validate = ( comm.Get_rank() == 0 )
        # Get the worker ID
        ranks_workergroup = get_worker_ranks( comm, self.num_masters )
        if not self.is_master:
            self.worker_id = ranks_workergroup.index( comm.Get_rank() )

    def make_comm_single(self,comm):
        """Create MPI communicator (Case 2): Rank 0 is master, all others are workers
            Sets is_master and worker_id attributes"""
        self.comm_block = comm
        self.is_master = ( self.comm_block.Get_rank() == 0 )
        self.should_validate = self.is_master
        if not self.is_master:
            self.worker_id = self.comm_block.Get_rank() - 1

    def free_comms(self):
        """Free active MPI communicators"""
        if self.comm_block is not None:
            self.comm_block.Free()
        if self.comm_masters is not None:
            self.comm_masters.Free()
        

class MPIKFoldManager(MPIManager):
    def __init__( self, NFolds, comm, data, algo, model_builder, num_epochs, train_list, 
                  val_list, num_masters=1, synchronous=False, callbacks=[], 
                  worker_callbacks=[], verbose=False, custom_objects={}):
        self.comm_world = comm
        self.comm_fold = None
        if NFolds == 1:
            ## make a regular MPIManager
            self.manager = MPIManager(comm, data, algo, model_builder, num_epochs, train_list,
                                      val_list, num_masters,
                                      synchronous, callbacks,
                                      worker_callbacks, verbose, custom_objects)
            return
        
        if int(comm.Get_size() / float(NFolds))<=1:
            print ("There is less than one master+one worker per fold, this isnt' going to work")
            
        ## actually split further the work in folds
        rank = comm.Get_rank()
        fold_num = int(rank * NFolds / comm.Get_size())
        self.comm_fold = comm.Split(fold_num)
        print ("For node {}, with block rank {}, send in fold {}".format(MPI.COMM_WORLD.Get_rank(), rank, fold_num))
        self.manager = None

        if val_list:
            print ("MPIKFoldManager would not expect to be given a validation list")
        all_files = train_list+val_list
        from sklearn.model_selection import KFold
        folding = KFold(n_splits = NFolds)
        folds = list(folding.split( all_files ))
        train, test = folds[ fold_num ]
        train_list_on_fold = list(np.asarray(all_files)[ train ])
        val_list_on_fold = list(np.asarray(all_files)[ test ])
        self.manager = MPIManager(self.comm_fold, data, algo, model_builder, num_epochs, train_list_on_fold,
                                  val_list_on_fold, num_masters, synchronous, callbacks,
                                  worker_callbacks, verbose, custom_objects)
                
    def train(self):
        self.manager.train()
    
    def figure_of_merit(self):
        fom = self.manager.figure_of_merit()
        if self.comm_fold is not None:
            foms = self.comm_world.allgather( fom )
            # filter out the None values
            foms = list(filter( None, foms))
            ## make the average and rms
            avg_fom = np.mean( foms )
            std_fom = np.std( foms )
            if self.comm_fold.Get_rank()==0:
                print ("Figure of merits over {} folds is {}+/-{}".format( len(foms), avg_fom, std_fom))
            return avg_fom
        else:
            if fom is not None:
                print ("Figure of merits from single value {}".format( fom ))
            return fom
