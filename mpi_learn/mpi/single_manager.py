from __future__ import division
from .manager import MPIManager

class MPISingleManager(MPIManager):
    """Topology with a single process. Check MPIManager for arguments"""

    def make_comms(self, comm):
        if comm.Get_size() != 1: print ("Warn: Several single processes are running")
        self.parent_rank = 0
        self.is_master = False
        n_instances = 1

        self.set_val_data()
        self.set_train_data()

        from .single_process import MPISingleWorker

        self.process = MPISingleWorker(data=self.data, algo=self.algo,
                                      model_builder=self.model_builder,
                                      num_epochs=self.num_epochs,
                                      verbose=self.verbose,
                                      monitor=self.monitor,
                                      custom_objects=self.custom_objects,
                                      early_stopping = self.early_stopping,
                                      target_metric = self.target_metric
                                    )

    def free_comms(self):
        pass

    def set_train_data(self):
        """Sets the training data files to be used by the current process - all of them"""
        print ("number of files",len(self.train_list))

        if not self.train_list:
            ## this is bad and needs to make it abort
            print ("There are no files for training, this is a fatal issue")
            import sys
            sys.exit(13)
            
        for f in self.train_list:
            print ("  {0}".format(f))
        self.data.set_file_names( self.train_list )

