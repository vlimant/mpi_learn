### ModelBuilder class and associated helper methods

from mpi_learn.utils import load_model

class ModelBuilder(object):
    """Class containing instructions for building neural net models.
        Derived classes should implement the build_model function.

        Attributes:
            comm: MPI communicator containing all running MPI processes
    """

    def __init__(self, comm):
        """Arguments:
            comm: MPI communicator 
        """
        self.comm = comm

    def build_model(self):
        """Should return an uncompiled Keras model."""
        raise NotImplementedError

class ModelFromJson(ModelBuilder):
    """ModelBuilder class that builds from model architecture specified
        in a JSON file.
        Attributes:
            filename: path to JSON file specifying model architecture
    """

    def __init__(self, comm, filename=None,json_str=None, custom_objects=None):
        self.filename = filename
        self.json_str = json_str
	self.custom_objects = custom_objects
        super(ModelFromJson, self).__init__(comm)

    def build_model(self):
        return load_model(filename=self.filename, json_str=self.json_str, custom_objects=self.custom_objects)
