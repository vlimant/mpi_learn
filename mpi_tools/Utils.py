### Utilities for mpi_tools module

import numpy as np

class Error(Exception):
    pass

def weights_from_shapes(weights_shapes):
    """Returns a list of numpy arrays representing the NN architecture"""
    return [ np.zeros( shape, dtype=np.float32 ) for shape in weights_shapes ]

def shapes_from_weights(weights):
    """Returns a list of tuples indicating the array shape of each layer of the NN"""
    return [ w.shape for w in weights ]

