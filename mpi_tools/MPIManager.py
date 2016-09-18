### MPIManager class 

from __future__ import division
from MPIProcess import MPIWorker, MPIMaster

class MPIManager(object):
    """The MPIManager class defines the topology of the MPI process network
        and creates master and worker objects for each process accordingly.

        Two configurations are available:
          1) one master supervising other masters, each controlling some workers
          2) one master and N-1 workers, where N is the number of MPI processes

        Attributes:
          num_masters: integer indicating the number of master processes.  
            If num_masters > 1, an additional master will be created to supervise all masters.
          train_steps: integer indicateing the number of steps each worker should train for.
          comm_block: MPI intracommunicator used for message passing between master and workers.
            Process 0 is the master and the other processes are workers.  
          comm_masters: MPI intracommunicator used for message passing between masters.
            (It will be None if there is only one master.)
    """

    def __init__(self, comm, num_masters, train_steps):
        """Create MPI communicator(s) needed for training, and create worker 
            or master object as appropriate.

            Params: 
            comm: MPI intracommunicator containing all processes
            num_masters: number of master processes
            train_steps: number of steps to train for
        """
        self.num_masters = num_masters
        self.train_steps = train_steps
        self.comm_block = None
        self.comm_masters = None
        self.make_comms(comm)

    def make_comms(self,comm):
        """Define the network topology by creating communicators linking masters with their slaves.
            Set comm_block to contain one master and all of its workers.
            Set comm_masters to contain all masters, including the "super-master" supervising them.
            Define a master or worker object as appropriate for each process.
        """
        # Case (1)
        if self.num_masters > 1:
            self.make_comms_many(comm)
            if self.is_master:
                if self.comm_masters.Get_rank() == 0: # rank 0 is the super-master
                    self.process = MPIMaster( self.comm_masters, parent_rank=None, 
                            child_comm=self.comm_masters )
                else:
                    self.process = MPIMaster( self.comm_masters, parent_rank=0, 
                            child_comm=self.comm_block )
            else:
                self.process = MPIWorker( self.comm_block, parent_rank=0, 
                        train_steps=self.train_steps )
        # Case (2)
        else:
            self.make_comm_single(comm)
            if self.is_master:
                self.process = MPIMaster( self.comm_block, parent_rank=None, child_comm=self.comm_block )
            else:
                self.process = MPIWorker( self.comm_block, parent_rank=0, train_steps=self.train_steps )

    def make_comms_many(self,comm):
        """Create MPI communicators (Case 1):
            Rank 0 of comm_block is the master, other ranks are workers.
            Rank 0 of comm_master is the super-master, other ranks are sub-masters.
            is_master is defined according to whether this process is a master or a worker."""

        # Create a communicator containing all processes except the first.  
        # Then divide that communicator into blocks, each with one master
        ranks_excludefirstprocess = range(1,comm.Get_size()) 
        comm_excludefirstprocess = comm.Create( comm.Get_group().Incl( ranks_excludefirstprocess ) )
        if comm.Get_rank() in ranks_excludefirstprocess:
            color_block = comm_excludefirstprocess.Get_rank() // self.num_masters 
            self.comm_block = comm_excludefirstprocess.Split( color_block )
            comm_excludefirstprocess.Free()
        else:
            self.comm_block = None
        # Create a communicator containing all masters
        ranks_mastergroup = [0]+range(1, comm.Get_size(), (comm.Get_size()-1) // self.num_masters)
        self.comm_masters = comm.Create( comm.Get_group().Incl(ranks_mastergroup) )
        self.is_master = ( comm.Get_rank() in ranks_mastergroup )

    def make_comm_single(self,comm):
        """Create MPI communicator (Case 2):
             Rank 0 is master, all others are workers"""
        self.comm_block = comm.Dup()
        self.is_master = ( self.comm_block.Get_rank() == 0 )

    def free_comms(self):
        """Free active MPI communicators"""
        if self.comm_block is not None:
            self.comm_block.Free()
        if self.comm_masters is not None:
            self.comm_masters.Free()
        
