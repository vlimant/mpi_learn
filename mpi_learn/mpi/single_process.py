import os,sys,json
import numpy as np
import socket
import time

from ..train.monitor import Monitor
from ..utils import Error, weights_from_shapes, shapes_from_weights
from .process import MPIWorker

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

        print ("MPIWorker {0} signing off".format(self.ranks))
        if self.monitor:
            self.update_monitor( self.monitor.get_stats() )        

        self.data.finalize()

    def validate(self):
        """Compute the loss on the validation data.
            Return a dictionary of validation metrics.
            Shamelessly copied from MPIMaster"""
        #return MPIMaster.validate(self)

        tell = True
        if self.has_parent:
            return {}
        #self.model.set_weights(self.weights)
        if tell: print ("Starting validation")
        val_metrics = np.zeros((1,))
        i_batch = 0
        for i_batch, batch in enumerate(self.data.generate_data()):
            new_val_metrics =  self.model.test_on_batch(x=batch[0], y =batch[1] )
            if val_metrics.shape != new_val_metrics.shape:
                val_metrics =  np.zeros(new_val_metrics.shape)
            val_metrics += new_val_metrics
            if self._short_batches and i_batch>self._short_batches: break
        val_metrics = val_metrics / float(i_batch+1)
        l = self.model.get_logs(val_metrics, val=True)
        self.update_history( l )

        if self.target_metric:
            m,opp,v = self.target_metric
            model,m= m.split(':') if ':' in m else (None,m)
            ##print ("target validation",model,m,opp,v)
            r = self.history_key()
            use = self.histories[r].get(model,None) if model else self.histories[r]
            if use:
                if m in use and ((opp=='>' and use[m][-1]>v) or (opp=='<' and use[m][-1]<v)):
                    print ("metric",m,"is",opp,v,"stop training")
                    self.stop_training = True
            else:
                print ("fatal target stopping cannot get",m)
                import sys
                sys.exit(14)
        if self.patience:
            m,opp,p = self.patience
            p = int(p)
            model,m=m.split(':') if ':' in m else (None,m)
            ##print ("patience validation",model,m,opp,p)
            r = self.history_key()            
            use = self.histories[r].get(model,None) if model else self.histories[r]
            ref = None
            smooth = False
            if use:
                if m in use:
                    ##print ("using",use[m])
                    if '~' in opp:
                        smooth = True
                        opp = opp.replace('~','')
                        ## do the averaging
                        if len(use[m])>=(2*p):
                            ref = np.mean(use[m][-(2*p):-p])
                            current = np.mean(use[m][-p:])
                        else:
                            if len(use[m])>=p:
                                ref = use[m][-p]
                                current = use[m][-1]
                    if ref is not None and current is not None and ((ref<current and opp=='<') or (ref>current and opp=='>')):
                        print ("metric",m,"is over",p,"patience boundary:",ref,"(ref)",opp,current,"(current)","with smoothing" if smooth else "")
                        self.stop_training = True
                else:
                    print ("fatal early stopping cannot get",m)
                    import sys
                    sys.exit(14)                    
            else:
                print ("fatal early stopping cannot get",m)
                import sys
                sys.exit(14)                
        print ("Validation metrics:")
        self.print_metrics(val_metrics)
        if tell: print ("Ending validation")
        return None

