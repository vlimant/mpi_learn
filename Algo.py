import numpy as np

from mpi_tools.Utils import weights_from_shapes

class Algo(object):
    """Base class for optimization algorithms.
        Attributes:
          loss: name of loss function (string)
          learning_rate: learning rate parameter for SGD
    """

    def __init__(self, loss, learning_rate):
        self.loss = loss
        self.learning_rate = learning_rate

    def apply_update(self, weights, gradient):
        raise NotImplementedError

class VanillaSGD(Algo):
    """Stochastic gradient descent with no extra frills."""
    
    def __init__(self, loss, learning_rate):
        super(VanillaSGD, self).__init__(loss, learning_rate)

    def apply_update(self, weights, gradient):
        """Move weights in the direction of the gradient, by the amount of the 
            learning rate."""
        new_weights = []
        for i,w in enumerate(weights):
            new_weights.append(np.subtract(w, self.learning_rate*gradient[i]))
        return new_weights
