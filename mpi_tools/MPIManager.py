### MPIManager class manages the creation of MPIProcesses
from __future__ import division

from MPIProcess import MPIWorker, MPIMaster

class MPIManager(object):
    def __init__(self, comm, num_masters, train_steps):
        self.num_masters = num_masters
        self.train_steps = train_steps
        self.comm_block = None
        self.comm_masters = None
        self.make_comms(comm)

    def make_comms(self,comm):
        if self.num_masters > 1:
            self.make_comms_many(comm)
            #Make Worker or Master, as needed
            if self.is_master:
                if self.comm_masters.Get_rank() == 0: # super-master
                    self.process = MPIMaster( self.comm_masters, parent_rank=None, child_comm=self.comm_masters )
                else:
                    self.process = MPIMaster( self.comm_masters, parent_rank=0, child_comm=self.comm_block )
            else:
                self.process = MPIWorker( self.comm_block, parent_rank=0, train_steps=self.train_steps )
        else:
            #If there is only one master, only one communicator is needed
            self.make_comm_single(comm)
            #Make Worker or Master, as needed
            if self.is_master:
                self.process = MPIMaster( self.comm_block, parent_rank=None, child_comm=self.comm_block )
            else:
                self.process = MPIWorker( self.comm_block, parent_rank=0, train_steps=self.train_steps )

    def make_comm_single(self,comm):
        '''Create communicator (only one master)'''
        self.comm_block = comm.Dup()
        #Rank 0 is master, all others are workers
        self.is_master = ( self.comm_block.Get_rank() == 0 )

    def make_comms_many(self,comm):
        '''Create a communicator for each master and a communicator containing all masters'''
        #If there are multiple masters, set aside one process to manage all masters
        ranks_removefirstprocess = range(1,comm.Get_size()) #all but first
        comm_removefirstprocess = comm.Create( comm.Get_group().Incl( ranks_removefirstprocess ) )
        #divide processes into num_masters groups (process 0 is not in any group)
        if comm.Get_rank() in ranks_removefirstprocess:
            color_block = comm_removefirstprocess.Get_rank() // self.num_masters 
            self.comm_block = comm_removefirstprocess.Split( color_block )
            #delete tmp communicator
            comm_removefirstprocess.Free()
        else:
            self.comm_block = None
        #make group of all masters
        ranks_mastergroup = [0]+range(1, comm.Get_size(), (comm.Get_size()-1) // self.num_masters)
        self.comm_masters = comm.Create( comm.Get_group().Incl(ranks_mastergroup) )
        self.is_master = ( comm.Get_rank() in ranks_mastergroup )

    def free_comms(self):
        if self.comm_block is not None:
            self.comm_block.Free()
        if self.comm_masters is not None:
            self.comm_masters.Free()
        
