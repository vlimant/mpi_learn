### Optimizers used to update master process weights

import numpy as np

from ..utils import weights_from_shapes

class Optimizer(object):
    """Base class for optimization algorithms.
        Currently doesn't do anything."""

    def __init__(self):
        pass

    def apply_update(self, weights, gradient):
        raise NotImplementedError


class VanillaSGD(Optimizer):
    """Stochastic gradient descent with no extra frills.
          learning_rate: learning rate parameter for SGD"""
    
    def __init__(self, learning_rate=0.01):
        super(VanillaSGD, self).__init__()
        self.learning_rate = learning_rate

    def apply_update(self, weights, gradient):
        """Move weights in the direction of the gradient, by the amount of the 
            learning rate."""
        new_weights = []
        for w, g in zip(weights, gradient):
            new_weights.append(np.subtract(w, self.learning_rate*g))
        return new_weights


class RunningAverageOptimizer(Optimizer):
    """Base class for AdaDelta, Adam, and RMSProp optimizers.
        rho (tunable parameter): decay constant used to compute running parameter averages
        epsilon (tunable parameter): small constant used to prevent division by zero
        running_g2: running average of the squared gradient, where squaring is done componentwise"""

    def __init__(self, rho=0.95, epsilon=1e-8):
        super(RunningAverageOptimizer, self).__init__()
        self.rho = rho
        self.epsilon = epsilon
        self.running_g2 = None

    def running_average_square_np(self, previous, update):
        """Computes and returns the running average of the square of a numpy array.
            previous (numpy array): value of the running average in the previous step
            update (numpy array): amount of the update"""
        new_contribution = (1-self.rho) * np.square(update)
        old_contribution = self.rho * previous
        return new_contribution + old_contribution

    def running_average_square(self, previous, update):
        """Returns the running average of the square of a quantity.
            previous (list of numpy arrays): value of the running average in the previous step
            update (list of numpy arrays): amount of the update"""
        if previous == 0:
            previous = [ np.zeros_like(u) for u in update ]
        result = []
        for prev, up in zip(previous, update):
            result.append( self.running_average_square_np( prev, up ) )
        return result

    def sqrt_plus_epsilon(self, value):
        """Computes running RMS from the running average of squares.
            value: numpy array containing the running average of squares"""
        return np.sqrt( np.add(value, self.epsilon) )


class Adam(RunningAverageOptimizer):
    """Adam optimizer.
        Note that the beta_2 parameter is stored internally as 'rho' 
        and "v" in the algorithm is called "running_g2"
        for consistency with the other running-average optimizers
        Attributes:
          learning_rate: base learning rate
          beta_1: decay rate for the first moment estimate
          m: running average of the first moment of the gradient
          t: time step
        """

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
            epsilon=1e-8):
        super(Adam, self).__init__(rho=beta_2, epsilon=epsilon)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.t = 0
        self.m = None

    def running_average_np(self, previous, update):
        """Computes and returns the running average of a numpy array.
            Parameters are the same as those for running_average_square_np"""
        new_contribution = (1-self.beta_1) * update
        old_contribution = self.beta_1 * previous
        return new_contribution + old_contribution

    def running_average(self, previous, update):
        """Returns the running average of the square of a quantity.
            Parameters are the same as those for running_average_square_np"""
        result = []
        for prev, up in zip(previous, update):
            result.append( self.running_average_np( prev, up ) )
        return result

    def apply_update(self, weights, gradient):
        """Update the running averages of the first and second moments
            of the gradient, and compute the update for this time step"""
        if self.running_g2 is None:
            self.running_g2 = [ np.zeros_like(g) for g in gradient ]
        if self.m is None:
            self.m = [ np.zeros_like(g) for g in gradient ]

        self.t += 1
        self.m = self.running_average( self.m, gradient )
        self.running_g2 = self.running_average_square( self.running_g2, gradient )
        alpha_t = self.learning_rate * (1 - self.rho**self.t)**(0.5) / (1 - self.beta_1**self.t)
        new_weights = []
        for w, g, g2 in zip(weights, self.m, self.running_g2):
            update = alpha_t * g / ( np.sqrt(g2) + self.epsilon )
            new_weights.append( w - update )
        return new_weights

class AdaDelta(RunningAverageOptimizer):
    """ADADELTA adaptive learning rate method.
        running_dx2: running average of squared parameter updates
        """

    def __init__(self, rho=0.95, epsilon=1e-8):
        super(AdaDelta, self).__init__(rho, epsilon)
        self.running_dx2 = None

    def apply_update(self, weights, gradient):
        """Update the running averages of gradients and weight updates,
            and compute the Adadelta update for this step."""
        if self.running_g2 is None:
            self.running_g2 = [ np.zeros_like(g) for g in gradient ]
        if self.running_dx2 is None:
            self.running_dx2 = [ np.zeros_like(g) for g in gradient ]

        self.running_g2 = self.running_average_square( self.running_g2, gradient )
        new_weights = []
        updates = []
        for w, g, g2, dx2 in zip(weights, gradient, self.running_g2, self.running_dx2):
            update = np.multiply( np.divide( self.sqrt_plus_epsilon(dx2), self.sqrt_plus_epsilon(g2) ), g )
            new_weights.append( np.subtract( w, update ) )
            updates.append(update)
        self.running_dx2 = self.running_average_square( self.running_dx2, updates )
        return new_weights

class RMSProp(RunningAverageOptimizer):
    """RMSProp adaptive learning rate method.
        learning_rate: base learning rate, kept constant
        """

    def __init__(self, rho=0.9, epsilon=1e-8, learning_rate=0.001):
        super(RMSProp, self).__init__(rho, epsilon)
        self.learning_rate = learning_rate

    def apply_update(self, weights, gradient):
        """Update the running averages of gradients,
            and compute the update for this step."""
        if self.running_g2 is None:
            self.running_g2 = [ np.zeros_like(g) for g in gradient ]

        self.running_g2 = self.running_average_square( self.running_g2, gradient )
        new_weights = []
        for w, g, g2 in zip(weights, gradient, self.running_g2):
            update = np.multiply( np.divide( self.learning_rate, self.sqrt_plus_epsilon(g2) ), g )
            new_weights.append( np.subtract( w, update ) )
        return new_weights

def get_optimizer(name):
    """Get optimizer class by string identifier"""
    lookup = {
            'sgd':      VanillaSGD,
            'adadelta': AdaDelta,
            'rmsprop':  RMSProp,
            'adam':     Adam,
            }
    return lookup[name]
