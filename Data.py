### Data class

import numpy as np
import h5py

def data_class_getter(name):
    """Returns the specified Data class"""
    data_dict = {
            "H5Data":H5Data,
            }
    try:
        return data_dict[name]
    except KeyError:
        print "%s is not a known Data class. Returning None..." % name
        return None


class Data(object):
    """Class providing an interface to the input training data.
        Derived classes should implement the load_data function.

        Attributes:
          file_names: list of data files to use for training
          batch_size: size of training batches
          val_samples: number of samples to be used in each validation
    """

    def __init__(self, file_names, batch_size, val_samples):
        """Stores the batch size and the names of the data files to be read.
            Params:
              file_names: list of data file names
              batch_size: batch size for training
              val_samples: number of samples to be used in each validation
        """
        self.file_names = file_names
        self.batch_size = batch_size
        self.val_samples = val_samples
            
    def generate_data(self):
       """Yields batches of training data until none are left."""
       for cur_file_name in self.file_names:
           cur_file_features, cur_file_labels = self.load_data(cur_file_name)
           num_in_file = cur_file_features.shape[0]

           # We get all available batches in this file, then move on to the next file.
           # If the batch size does not evently divide the number of examples per file,
           # then some examples at the end of each file will go unused.  
           for cur_pos in range(0, num_in_file, self.batch_size):
               next_pos = cur_pos + self.batch_size 
               if next_pos < num_in_file:
                   yield ( cur_file_features[cur_pos:next_pos],
                           cur_file_labels[cur_pos:next_pos] )

    def load_data(self, in_file):
        """Input: name of file from which the data should be loaded
            Returns: tuple (X,Y) where X and Y are numpy arrays containing features 
                and labels, respectively, for all data in the file

            Not implemented in base class; derived classes should implement this function"""
        raise NotImplementedError

class H5Data(Data):
    """Loads data stored in hdf5 files
        Attributes:
          features_name, labels_name: names of the datasets containing the features
          and labels, respectively
    """

    def __init__(self, file_names, batch_size, val_samples,
            features_name='features', labels_name='labels'):
        """Initializes and stores names of feature and label datasets"""
        super(H5Data, self).__init__(file_names, batch_size, val_samples)
        self.features_name = features_name
        self.labels_name = labels_name

    def load_data(self, in_file_name):
        """Loads numpy arrays from H5 file"""
        h5_file = h5py.File( in_file_name, 'r' )
        X = h5_file[self.features_name][:]
        Y = h5_file[self.labels_name][:]
        h5_file.close()
        return X,Y 
