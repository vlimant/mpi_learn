### ModelBuilder class and associated helper methods

from mpi_learn.utils import load_model, get_device_name

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

    def __init__(self, comm, filename=None,json_str=None, custom_objects={}, weights=None):
        self.filename = filename
        self.json_str = json_str
        self.weights = weights
	self.custom_objects = custom_objects
        super(ModelFromJson, self).__init__(comm)

    def build_model(self):
        return load_model(filename=self.filename, json_str=self.json_str, custom_objects=self.custom_objects, weights_file=self.weights)

class ModelFromJsonTF(ModelBuilder):
    """ModelBuilder class that builds from model architecture specified
        in a JSON file. Uses Tensorflow and builds the model on the 
        specified GPU.
        Attributes:
            filename: path to JSON file specifying model architecture
            device: name of the device to use (ex: "/gpu:2")
    """

    def __init__(self, comm, filename=None, json_str=None, device_name='cpu', 
            custom_objects={}, weights=None):
        self.filename = filename
        self.json_str = json_str
        self.weights = weights
	self.custom_objects = custom_objects
        self.device = self.get_device_name(device_name)
        super(ModelFromJsonTF, self).__init__(comm)

    def get_device_name(self, device):
        """Returns a TF-style device identifier for the specified device.
            input: 'cpu' for CPU and 'gpuN' for the Nth GPU on the host"""
        if device == 'cpu':
            dev_num = 0
            dev_type = 'cpu'
        elif device.startswith('gpu'):
            try:
                dev_num = int(device[3:])
                dev_type = 'gpu'
            except ValueError:
                print "GPU number could not be parsed from {}; using CPU".format(device)
                dev_num = 0
                dev_type = 'cpu'
        else:
            print "Please specify 'cpu' or 'gpuN' for device name"
            dev_num = 0
            dev_type = 'cpu'
        return get_device_name(dev_type, dev_num, backend='tensorflow')

    def build_model(self):
        import keras.backend as K
        K.set_session( K.tf.Session( config=K.tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False,
            gpu_options=K.tf.GPUOptions(
                per_process_gpu_memory_fraction=1./self.comm.Get_size()) ) ) )
        with K.tf.device(self.device):
            model = load_model(filename=self.filename, json_str=self.json_str, 
                    custom_objects=self.custom_objects, weights_file=self.weights)
        return model
