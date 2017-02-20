### Data class and associated helper methods

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
        print ("{0:s} is not a known Data class. Returning None...".format(name))
        return None


class Data(object):
    """Class providing an interface to the input training data.
        Derived classes should implement the load_data function.

        Attributes:
          file_names: list of data files to use for training
          batch_size: size of training batches
    """

    def __init__(self, batch_size):
        """Stores the batch size and the names of the data files to be read.
            Params:
              batch_size: batch size for training
        """
        self.batch_size = batch_size

    def set_file_names(self, file_names):
        self.file_names = file_names
            
    def generate_data(self):
       """Yields batches of training data until none are left."""
       leftovers = None
       for cur_file_name in self.file_names:
           cur_file_features, cur_file_labels = self.load_data(cur_file_name)
           # concatenate any leftover data from the previous file
           if leftovers is not None:
               cur_file_features = self.concat_data( leftovers[0], cur_file_features )
               cur_file_labels = self.concat_data( leftovers[1], cur_file_labels )
               leftovers = None
           num_in_file = self.get_num_samples( cur_file_features )

           for cur_pos in range(0, num_in_file, self.batch_size):
               next_pos = cur_pos + self.batch_size 
               if next_pos <= num_in_file:
                   yield ( self.get_batch( cur_file_features, cur_pos, next_pos ),
                           self.get_batch( cur_file_labels, cur_pos, next_pos ) )
               else:
                   leftovers = ( self.get_batch( cur_file_features, cur_pos, num_in_file ),
                                 self.get_batch( cur_file_labels, cur_pos, num_in_file ) )

    def count_data(self):
        """Counts the number of data points across all files"""
        num_data = 0
        for cur_file_name in self.file_names:
            cur_file_features, cur_file_labels = self.load_data(cur_file_name)
            num_data += self.get_num_samples( cur_file_features )
        return num_data

    def is_numpy_array(self, data):
        return isinstance( data, np.ndarray )

    def get_batch(self, data, start_pos, end_pos):
        """Input: a numpy array or list of numpy arrays.
            Gets elements between start_pos and end_pos in each array"""
        if self.is_numpy_array(data):
            return data[start_pos:end_pos] 
        else:
            return [ arr[start_pos:end_pos] for arr in data ]

    def concat_data(self, data1, data2):
        """Input: data1 as numpy array or list of numpy arrays.  data2 in the same format.
           Returns: numpy array or list of arrays, in which each array in data1 has been
             concatenated with the corresponding array in data2"""
        if self.is_numpy_array(data1):
            return np.concatenate( (data1, data2) )
        else:
            return [ self.concat_data( d1, d2 ) for d1,d2 in zip(data1,data2) ]

    def get_num_samples(self, data):
        """Input: dataset consisting of a numpy array or list of numpy arrays.
            Output: number of samples in the dataset"""
        if self.is_numpy_array(data):
            return len(data)
        else:
            return len(data[0])

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

    def __init__(self, batch_size, 
            features_name='features', labels_name='labels'):
        """Initializes and stores names of feature and label datasets"""
        super(H5Data, self).__init__(batch_size)
        self.features_name = features_name
        self.labels_name = labels_name

    def load_data(self, in_file_name):
        """Loads numpy arrays from H5 file.
            If the features/labels groups contain more than one dataset,
            we load them all, alphabetically by key."""
        h5_file = h5py.File( in_file_name, 'r' )
        X = self.load_hdf5_data( h5_file[self.features_name] )
        Y = self.load_hdf5_data( h5_file[self.labels_name] )
        h5_file.close()
        return X,Y 

    def load_hdf5_data(self, data):
        """Returns a numpy array or (possibly nested) list of numpy arrays 
            corresponding to the group structure of the input HDF5 data.
            If a group has more than one key, we give its datasets alphabetically by key"""
        if hasattr(data, 'keys'):
            out = [ self.load_hdf5_data( data[key] ) for key in sorted(data.keys()) ]
        else:
            out = data[:]
        return out

    def count_data(self):
        """This is faster than using the parent count_data
            because the datasets do not have to be loaded
            as numpy arrays"""
        num_data = 0
        for in_file_name in self.file_names:
            h5_file = h5py.File( in_file_name, 'r' )
            X = h5_file[self.features_name]
            if hasattr(X, 'keys'):
                num_data += len(X[ X.keys()[0] ])
            else:
                num_data += len(X)
            h5_file.close()
        return num_data
