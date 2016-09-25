import numpy as np

from mpi_tools.Utils import weights_from_shapes

class Algo(object):
    """Base class for optimization algorithms.
        Attributes:
          loss: name of loss function (string)
          validate_every: number of updates to wait between validations
    """

    def __init__(self, loss, validate_every):
        self.loss = loss
        self.validate_every = validate_every

    def apply_update(self, weights, gradient):
        raise NotImplementedError

class VanillaSGD(Algo):
    """Stochastic gradient descent with no extra frills.
          learning_rate: learning rate parameter for SGD"""
    
    def __init__(self, loss, validate_every, learning_rate):
        super(VanillaSGD, self).__init__(loss, validate_every)
        self.learning_rate = learning_rate

    def apply_update(self, weights, gradient):
        """Move weights in the direction of the gradient, by the amount of the 
            learning rate."""
        new_weights = []
        for i,w in enumerate(weights):
            new_weights.append(np.subtract(w, self.learning_rate*gradient[i]))
        return new_weights

class AdaDelta(Algo):
    """ADADELTA adaptive learning rate method.
        rho (tunable parameter): decay constant used to compute running parameter averages
        epsilon (tunable parameter): small constant used to prevent division by zero
        running_g2: running average of the squared gradient, where squaring is done componentwise
        running_dx2: running average of squared parameter updates
        """

    def __init__(self, loss, validate_every, rho=0.95, epsilon=1e-8):
        super(AdaDelta, self).__init__(loss, validate_every)
        self.rho = rho
        self.epsilon = epsilon
        self.running_g2 = None
        self.running_dx2 = None

    def running_average_np(self, previous, update):
        """Computes and returns the running average of the square of a numpy array.
            previous (numpy array): value of the running average in the previous step
            update (numpy array): amount of the update"""
        new_contribution = (1-self.rho) * np.square(update)
        old_contribution = self.rho * previous
        return new_contribution + old_contribution

    def running_average(self, previous, update):
        """Returns the running average of the square of a quantity.
            previous (list of numpy arrays): value of the running average in the previous step
            update (list of numpy arrays): amount of the update"""
        if previous == 0:
            previous = [ np.zeros_like(u) for u in update ]
        result = []
        for i in range(len(update)):
            result.append( self.running_average_np( previous[i], update[i] ) )
        return result

    def rms(self, value):
        """Computes running RMS from the running average of squares.
            value: numpy array containing the running average of squares"""
        return np.sqrt( np.add(value, self.epsilon) )

    def apply_update(self, weights, gradient):
        """Update the running averages of gradients and weight updates,
            and compute the ADADELTA update for this step."""
        if self.running_g2 is None:
            self.running_g2 = [ np.zeros_like(g) for g in gradient ]
        if self.running_dx2 is None:
            self.running_dx2 = [ np.zeros_like(g) for g in gradient ]

        self.running_g2 = self.running_average( self.running_g2, gradient )
        new_weights = []
        updates = []
        for i in range(len(weights)):
            update = np.multiply( np.divide( self.rms( self.running_dx2[i] ), 
                self.rms( self.running_g2[i] ) ), gradient[i] )
            new_weights.append( np.subtract( weights[i], update ) )
            updates.append(update)
        self.running_dx2 = self.running_average( self.running_dx2, updates )
        return new_weights
