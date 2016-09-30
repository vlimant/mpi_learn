import numpy as np
from keras.optimizers import SGD

from Optimizer import get_optimizer

class Algo(object):
    """The Algo class contains all information about the training algorithm.
        Attributes:
          optimizer: instance of the Optimizer class used to compute training updates
          optimizer_name: name of the optimizer
          staleness: difference in time step between master and most recent worker's update
          loss: name of loss function (string)
          validate_every: number of updates to wait between validations
          """

    supported_opts = ['loss','validate_every']

    def __init__(self, optimizer, **kwargs):
        """optimizer: string naming an optimization algorithm as defined in Optimizer.get_optimizer()
            Configuration options should be provided as keyword arguments.
            Available arguments are:
               loss: string naming the loss function to be used for training
               validate_every: number of time steps to wait between validations
            Optimizer configuration options should be provided as additional
            named arguments (check your chosen optimizer class for details)."""
        for opt in self.supported_opts:
            setattr(self, opt, kwargs.get(opt))

        self.optimizer_name = optimizer
        optimizer_args = { arg:val for arg,val in kwargs.iteritems() 
            if arg not in self.supported_opts }
        self.optimizer = get_optimizer( optimizer )(**optimizer_args)

    def __str__(self):
        strs = [ opt+": "+str(getattr(self, opt)) for opt in self.supported_opts ]
        return '\n'.join(strs)

    ### For Worker ###

    def compile_model(self, model):
        """Compile the model. Workers are only responsible for computing the gradient and 
            sending it to the master, so we use ordinary SGD with learning rate 1 and 
            compute the gradient as (old weights - new weights) after each batch"""
        sgd = SGD(lr=1.0)
        model.compile( loss=self.loss, optimizer=sgd, metrics=['accuracy'] )

    def compute_update(self, cur_weights, new_weights):
        """Computes the update to be sent to the parent process"""
        update = []
        for cur_w, new_w in zip( cur_weights, new_weights ):
            update.append( np.subtract( cur_w, new_w ) )
        return update

    def worker_sends_weights_or_updates(self):
        """Returns 'weights' or 'gradient' according to the needs of 
            the training algorithm."""
        return 'update'

    def set_worker_model_weights(self, model, weights):
        """Apply a new set of weights to the worker's copy of the model"""
        model.set_weights( weights )

    ### For Master ###

    def apply_update(self, weights, update):
        """Calls the optimizer to apply an update
            and returns the resulting weights"""
        new_weights = self.optimizer.apply_update( weights, update )
        return new_weights

