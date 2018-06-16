### Optimizers used to update master process weights

import numpy as np
import copy
import pickle
import os

from ..utils import weights_from_shapes

class Optimizer(object):
    """Base class for optimization algorithms.
        Currently doesn't do anything."""

    def __init__(self):
        pass

    def reset(self):
        pass

    def apply_update(self, weights, gradient):
        raise NotImplementedError

    def save(self, fn = None):
        if fn is None:
            fn = 'master-opt-{}.algo'.format( os.getpid())
        d= open(fn,'wb')
        pickle.dump(self, d)
        d.close()

    def load(self, fn = 'algo_.pkl'):
        d = open(fn, 'rb')
        self = pickle.load( d )
        d.close()


class MultiOptimizer(Optimizer):
    def __init__(self, opt, s):
        self.opts = [copy.deepcopy(opt) for i in range(s)]

    def reset(self):
        for o in self.opts:
            o.reset()

    def apply_update(self, weights, gradient):
        r = []
        for o,w,g in zip(self.opts, weights, gradient):
            r.append( o.apply_update(w,g) )
        return r

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
            if type(w) == list:
                new_weights.append( [] )
                for ww, gg in zip(w,g):
                    new_weights[-1].append( np.subtract( ww, self.learning_rate*gg) )
            else:
                new_weights.append(np.subtract(w, self.learning_rate*g))
        return new_weights


class RunningAverageOptimizer(Optimizer):
    """Base class for AdaDelta, Adam, and RMSProp optimizers.
        rho (tunable parameter): decay constant used to compute running parameter averages
        epsilon (tunable parameter): small constant used to prevent division by zero
        running_g2: running average of the squared gradient, where squaring is done componentwise"""

    def __init__(self, rho=0.95, epsilon=1e-8):
        super(RunningAverageOptimizer, self).__init__()
        self.init_rho = rho
        self.init_epsilon = epsilon
        RunningAverageOptimizer.reset(self)

    def reset(self):
        super(RunningAverageOptimizer, self).reset()
        self.epsilon = self.init_epsilon
        self.rho = self.init_rho
        self.running_g2 = None

    def running_average_square_np(self, previous, update):
        """Computes and returns the running average of the square of a numpy array.
            previous (numpy array): value of the running average in the previous step
            update (numpy array): amount of the update"""
        try:
            new_contribution = (1-self.rho) * np.square(update)
            old_contribution = self.rho * previous
            return new_contribution + old_contribution
        except Exception as e:
            print ("FAILED TO COMPUTE THE RUNNING AVG SQUARE due to",str(e))
            print ("rho",self.rho)
            print ("min update",np.min(update))
            print ("max update",np.max(update))
            print ("min previous",np.min(previous))
            print ("max previous",np.max(previous))
            return previous

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
        self.init_learning_rate = learning_rate
        self.init_beta_1 = beta_1
        Adam.reset(self)

    def reset(self):
        super(Adam, self).reset()
        self.beta_1 = self.init_beta_1
        self.learning_rate = self.init_learning_rate
        self.t = 0
        self.m = None

    def running_average_np(self, previous, update):
        """Computes and returns the running average of a numpy array.
            Parameters are the same as those for running_average_square_np"""
        try:
            new_contribution = (1-self.beta_1) * update
            old_contribution = self.beta_1 * previous
            return new_contribution + old_contribution
        except Exception as e:
            print ("FAILED TO UPDATE THE RUNNING AVERAGE due to",str(e))
            print ("beta_1",self.beta_1)
            print ("min update",np.min(update))
            print ("max update",np.max(update))
            print ("min previous",np.min(previous))
            print ("max previous",np.max(previous))
            return previous


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
            try:
                update = alpha_t * g / ( np.sqrt(g2) + self.epsilon )
            except Exception as e:
                print ("FAILED TO MAKE A WEIGHT UPDATE due to",str(e))
                print ("alpha_t",alpha_t)
                print ("beta_1",self.beta_1)
                print ("t",self.t)
                print ("learning rate",self.learning_rate)
                print ("rho",self.rho)
                print ("epsilon",self.epsilon)
                print ("min gradient",np.min( g ))
                print ("max gradient",np.max( g ))
                print ("min gradient 2",np.min( g2 ))
                print ("max gradient 2",np.max( g2 ))
                try:
                    update = alpha_t * g / ( np.sqrt(g2) + self.epsilon )
                    print ("a")
                    try:
                        new_weights.append( w - update )
                    except:
                        print ("no sub")
                except:
                    try:
                        update = g / ( np.sqrt(g2) + self.epsilon )
                        print ("b")
                        print ("min b",np.min( update ))
                        print ("max b",np.max( update ))
                        print ("min |b|",np.min(np.fabs( update)))
                        #update *= alpha_t
                    except:
                        try:
                            update = 1./ ( np.sqrt(g2) + self.epsilon )
                            print ("c")
                        except:
                            try:
                                update = 1./ ( g2 + self.epsilon )
                                print ("d")
                            except:
                                print ("e")
                
                update = 0        
            new_weights.append( w - update )
        return new_weights

class AdaDelta(RunningAverageOptimizer):
    """ADADELTA adaptive learning rate method.
        running_dx2: running average of squared parameter updates
        """

    def __init__(self, rho=0.95, epsilon=1e-8):
        super(AdaDelta, self).__init__(rho, epsilon)
        AdaDelta.reset(self)

    def reset(self):
        super(AdaDelta, self).reset()
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
        self.init_learning_rate = learning_rate
    def reset(self):
        super(RMSProp, self).reset()
        self.learning_rate = self.init_learning_rate

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

class OptimizerBuilder(object):
    """Builds a new Keras or Torch optimizer and optionally wraps it in horovod DistributedOptimizer."""

    def __init__(self, name, config=None, horovod_wrapper=False):
        self.name = name
        self.config = config
        self.horovod_wrapper = horovod_wrapper

    def build(self):
        from keras.optimizers import deserialize
        if self.config is None:
            self.config = {}
        opt_config = {'class_name': self.name, 'config': self.config}
        opt = deserialize(opt_config)
        if self.horovod_wrapper:
            import horovod.keras as hvd
            if hasattr(opt, 'lr'):
                opt.lr *= hvd.size()
            opt = hvd.DistributedOptimizer(opt)
        return opt

    def build_torch(self, model):
        import torch
        opt = torch.optim.SGD(model.parameters(), 1.)
        if self.horovod_wrapper:
            import horovod.torch as hvd
            opt = hvd.DistributedOptimizer(opt, named_parameters=model.named_parameters())
        return opt
