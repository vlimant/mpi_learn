### Utilities for mpi_learn module

import numpy as np

class Error(Exception):
    pass

def weights_from_shapes(weights_shapes):
    """Returns a list of numpy arrays representing the NN architecture"""
    return [ np.zeros( shape, dtype=np.float32 ) for shape in weights_shapes ]

def shapes_from_weights(weights):
    """Returns a list of tuples indicating the array shape of each layer of the NN"""
    return [ w.shape for w in weights ]

def get_num_gpus():
    """Returns the number of GPUs available"""
    from pycuda import driver 
    driver.init()
    num_gpus = driver.Device.count()
    return num_gpus

def import_keras(tries=10):
    """There is an issue when multiple processes import Keras simultaneously --
        the file .keras/keras.json is sometimes not read correctly.  
        as a workaround, just try several times to import keras."""
    for try_num in range(tries):
        try:
            import keras
            return
        except ValueError:
            print "Unable to import keras. Trying again: %d" % try_num
            sleep(0.1)
    print "Failed to import keras!"

def load_model(arch_file, weights_file=None):
    """Loads model architecture from JSON and instantiates the model.
        arch_file: JSON file specifying model architecture
        weights_file: HDF5 file containing model weights"""
    import_keras()
    from keras.models import model_from_json
    with open( arch_file ) as arch_f:
        model = model_from_json( arch_f.readline() ) 
    if weights_file is not None:
        model.load_weights( weights_file )
    return model

