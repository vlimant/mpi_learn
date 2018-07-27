import os,sys,json
import numpy as np
import socket
import time

from ..train.monitor import Monitor
from ..utils import Error, weights_from_shapes, shapes_from_weights
from .process import MPIWorker, MPIMaster

class MPISingleWorker(MPIWorker):
    """This class trains its model with no communication to other processes"""
    def __init__(self, num_epochs, data, algo, model_builder,
                verbose, monitor, custom_objects,
                early_stopping, target_metric):

        self.time_step = 0
        self.has_parent = False

        self.best_val_loss = None
        self.target_metric = (target_metric if type(target_metric)==tuple else tuple(map(lambda s : float(s) if s.replace('.','').isdigit() else s, target_metric.split(',')))) if target_metric else None
        self.patience = (early_stopping if type(early_stopping)==tuple else tuple(map(lambda s : float(s) if s.replace('.','').isdigit() else s, early_stopping.split(',')))) if early_stopping else None

        self.num_epochs = num_epochs
        self.data = data
        self.algo = algo
        self.model_builder = model_builder
        self.verbose = verbose
        self.histories = {}
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
        self._short_batches = 0
        self._is_shadow = False

        self.monitor = Monitor() if monitor else None
        
        self.process_comm = None
        self.rank = 0
        self.ranks = "{0}:{1}:{2}".format(0, '-', '-')
        self.build_model()
        self.weights = self.model.get_weights()

    def train(self):
        self.check_sanity()

        for epoch in range(self.num_epochs):
            print ("MPISingle {0} beginning epoch {1:d}".format(self.ranks, epoch))
            if self.monitor:
                self.monitor.start_monitor()
            epoch_metrics = np.zeros((1,))
            i_batch = 0

            for i_batch, batch in enumerate(self.data.generate_data()):
                train_metrics = self.model.train_on_batch( x=batch[0], y=batch[1] )
                if epoch_metrics.shape != train_metrics.shape:
                    epoch_metrics = np.zeros( train_metrics.shape)
                epoch_metrics += train_metrics

                ######
                self.update = self.algo.compute_update( self.weights, self.model.get_weights() )
                self.weights = self.algo.apply_update( self.weights, self.update )
                self.algo.set_worker_model_weights( self.model, self.weights )
                ######

                if self._short_batches and i_batch>self._short_batches: break

            if self.monitor:
                self.monitor.stop_monitor()
            epoch_metrics = epoch_metrics / float(i_batch+1)
            l = self.model.get_logs( epoch_metrics )
            self.update_history( l )

            if self.stop_training:
                break

            self.validate()

        print ("MPISingle {0} signing off".format(self.ranks))
        if self.monitor:
            self.update_monitor( self.monitor.get_stats() )        

        self.data.finalize()

    def validate(self):
        """Compute the loss on the validation data.
            Return a dictionary of validation metrics.
            Shamelessly copied from MPIMaster"""
        return MPIMaster.validate(self)

