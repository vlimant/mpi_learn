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
    print ("Determining number of GPUs...")
    from pycuda import driver 
    driver.init()
    num_gpus = driver.Device.count()
    print ("Number of GPUs: {}".format(num_gpus))
    return num_gpus

def get_device_name(dev_type, dev_num, backend='theano'):
    """Returns cpu/gpu device name formatted for
    theano or keras, as specified by the backend argument"""
    if backend == 'tensorflow':
        return "/%s:%d" % (dev_type, dev_num)
    else:
        if dev_type == 'cpu':
            return 'cpu'
        else:
            return dev_type+str(dev_num)

def import_keras(tries=10):
    """There is an issue when multiple processes import Keras simultaneously --
        the file .keras/keras.json is sometimes not read correctly.  
        as a workaround, just try several times to import keras."""
    for try_num in range(tries):
        try:
            import keras
            return
        except ValueError:
            print ("Unable to import keras. Trying again: {0:d}".format(try_num))
            from time import sleep
            sleep(0.1)
    print ("Failed to import keras!")

def load_model(filename=None, json_str=None, weights_file=None, custom_objects={}):
    """Loads model architecture from JSON and instantiates the model.
        filename: path to JSON file specifying model architecture
        json_str: (or) a json string specifying the model architecture
        weights_file: path to HDF5 file containing model weights
	custom_objects: A Dictionary of custom classes used in the model keyed by name"""
    import_keras()
    from keras.models import model_from_json
    if filename != None:
        with open( filename ) as arch_f:
            json_str = arch_f.readline()
    model = model_from_json( json_str, custom_objects=custom_objects) 
    if weights_file is not None:
        model.load_weights( weights_file )
    return model

