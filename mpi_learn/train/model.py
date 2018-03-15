### ModelBuilder class and associated helper methods

from mpi_learn.utils import load_model, get_device_name

class MPIModel(object):
    """Class that abstract all details of the model
    """
    def __init__(self, model=None, models=None):
        """Arguments:                                                                                                                                                                                                                                                                            comm: MPI communicator                                                                                                                                                                                                                                                          """
        self.model = model
        self.models = models
        if model and models:
            raise Exception("Cannot specify single and multiple models")

    def get_weights(self):
        if self.model:
            return self.model.get_weights()
        else:
            l_weights = []
            for m in self.models:
                l_weights.append( m.get_weights() )
            return l_weights
        
    def set_weights(self, w ):
        if self.model:
            self.model.set_weights( w )
        else:
            for m,mw in zip(self.models, w ):
                m.set_weights( mw )
            
    def history(self):
        if self.model:
            return self.model.history

    def set_history(self, h):
        if self.model:
            self.model.history = h

    def compile(self, **args):
        if self.model:
            self.model.compile( **args )
        else:
            for m in self.models:
                m.compile( **args )

    def metrics_names(self):
        if self.model:
            return self.model.metrics_names

    def callback_model(self):
        if self.model:
            return getattr(self.model, 'callback_model', None)
        
    def train_on_batch(self, **args):
        if self.model:
            return self.model.train_on_batch( **args )
        else:
            for m in self.models:
                h = m.train_on_batch( **args )
            return h ## return the history of the last model, for lack of better idea for now
                
    def test_on_batch(self, **args):
        if self.model:
            return self.model.test_on_batch( **args )        

    def save(self, *args,**kwargs):
        if self.model:
            self.model.save( *args, **kwargs )
        else:
            for m in self.models:
                m.save( *args, **kwargs )


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
        if type(self.filename) == list:
            models = []
            for fn in self.filename:
                models.append(load_model(filename=fn))
            return MPIModel(models = models)
        else:        
            return MPIModel(model=load_model(filename=self.filename, json_str=self.json_str, custom_objects=self.custom_objects, weights_file=self.weights))

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
            if type(self.filename) == list:
                models = []
                for fn in self.filename:
                    models.append(load_model(filename=fn))
                return MPIModel(models = models)
            else:
                model = load_model(filename=self.filename, json_str=self.json_str, 
                                   custom_objects=self.custom_objects, weights_file=self.weights)
                return MPIModel(model = model)
