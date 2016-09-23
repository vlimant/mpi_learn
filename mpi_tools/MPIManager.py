### MPIManager class 

from __future__ import division

from Data import H5Data
from GPU import get_num_gpus

def get_master_ranks(comm, num_masters=1):
    """Arguments: 
        comm: MPI intracommunicator containing all processes
        num_masters: number of processes that will be assigned as masters
       Returns:
        a list of integers corresponding to the MPI ranks that will be assigned as masters"""
    if num_masters > 1:
        return [0]+range(1, comm.Get_size(), (comm.Get_size()-1) // num_masters)
    return [0]

def get_worker_ranks(comm, num_masters=1):
    """Arguments:
        comm: MPI intracommunicator containing all processes
        num_masters: number of processes that will be assigned as masters
       Returns:
        a list of integers corresponding to the MPI ranks that will be assigned as workers"""
    master_ranks = get_master_ranks( comm, num_masters )
    return [ x for x in range(comm.Get_size()) if x not in master_ranks ]

def get_device(comm, num_masters=1):
    rank = comm.Get_rank()
    worker_ranks = get_worker_ranks( comm, num_masters )
    if rank in worker_ranks:
        worker_id = worker_ranks.index( rank )
    else:
        worker_id = -1
    max_gpu = get_num_gpus() - 1
    if worker_id < 0 or worker_id > max_gpu:
        return 'cpu'
    else:
        return 'gpu%d' % worker_id

class MPIManager(object):
    """The MPIManager class defines the topology of the MPI process network
        and creates master and worker objects for each process accordingly.

        Two configurations are available:
          1) one master supervising other masters, each controlling some workers
          2) one master and N-1 workers, where N is the number of MPI processes

        Attributes:
          process: the MPI worker or master object running on this process
          num_masters: integer indicating the number of master processes.  
            If num_masters > 1, an additional master will be created to supervise all masters.
          num_workers: integer indicating the number of worker processes
          worker_id: ID of worker node, used for indexing training data files
          batch_size (integer): number of training examples workers process at once
          num_epochs (integer): number of times to iterate over the training data
          comm_block: MPI intracommunicator used for message passing between master and workers.
            Process 0 is the master and the other processes are workers.  
          comm_masters: MPI intracommunicator used for message passing between masters.
            (It will be None if there is only one master.)
          train_list: list of training data file names
          val_list: list of validation data file names
          is_master: boolean determining if this process is a master
          should_validate: boolean determining if this process should run training validation
    """

    def __init__(self, comm, batch_size, num_epochs, train_list, val_list, num_masters=1):
        """Create MPI communicator(s) needed for training, and create worker 
            or master object as appropriate.

            Params: 
            comm: MPI intracommunicator containing all processes
            num_masters: number of master processes
            batch_size: number of training examples to process at once
            num_epochs: number of times to iterate over the training data
            train_list: list of training data files
            val_list: list of validation data files
        """
        self.num_masters = num_masters
        self.num_workers = comm.Get_size() - self.num_masters 
        if self.num_masters > 1:
            self.num_workers -= 1 # one process is taken up by the super-master
        self.worker_id = -1

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.train_list = train_list
        self.val_list = val_list
        self.comm_block = None
        self.comm_masters = None
        self.is_master = None
        self.should_validate = None

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

        parent_rank = 0

        # Case (1)
        if self.num_masters > 1:
            self.make_comms_many(comm)
            if self.is_master:
                parent_comm = self.comm_masters
                if self.comm_masters.Get_rank() == 0: # rank 0 is the super-master
                    child_comm = self.comm_masters
                    parent_rank = None
                else:
                    child_comm = self.comm_block
        # Case (2)
        else:
            self.make_comm_single(comm)
            if self.is_master:
                parent_comm = self.comm_block
                child_comm = self.comm_block
                parent_rank = None

        # Process initialization
        from MPIProcess import MPIWorker, MPIMaster
        if self.is_master:
            val_data = self.make_val_data()
            self.process = MPIMaster( parent_comm, parent_rank=parent_rank, 
                    data=val_data, child_comm=child_comm )
        else:
            train_data = self.make_train_data()
            self.process = MPIWorker( parent_comm=self.comm_block, parent_rank=parent_rank, 
                    num_epochs=self.num_epochs, data=train_data )

    def make_train_data(self):
        """Creates and returns the train data object associated with the current MPI process"""
        files_per_worker = len(self.train_list) // self.num_workers
        files_for_this_worker = self.train_list[ 
                self.worker_id*files_per_worker : (self.worker_id+1)*files_per_worker ]
        # The worker takes an extra file if needed
        if self.worker_id < len(self.train_list) % self.num_workers:
            files_for_this_worker.append(self.train_list[ self.num_workers*files_per_worker + self.worker_id ])
        print "Files for worker %d:" % self.comm_block.Get_rank()
        for f in files_for_this_worker:
            print "  %s" % f
        return H5Data( files_for_this_worker, self.batch_size )

    def make_val_data(self):
        """Creates and returns the validation data object associated with the current MPI process
            (only the master process has validation data associated with it)"""
        if not self.should_validate: return None
        print "Files for validation:" 
        for f in self.val_list:
            print "  %s" % f
        return H5Data( self.val_list, self.batch_size )

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
        self.comm_block = comm.Dup()
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
        
