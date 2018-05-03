### ModelBuilder class and associated helper methods

from mpi_learn.utils import load_model, get_device_name
import numpy as np
import copy

class MPICallbacks(object):
    def __init__(self, cbks):
        self.cbks = cbks

        self.callbacks = None
        self.callbacksS = None

        self.callback_model = None
        self.callback_models = None
        
    def handle(self, modelO):
        import keras.callbacks as cbks
        if modelO.model:
            ## single model case
            self.callbacks = cbks.CallbackList( self.cbks + [modelO.model.history] )
            if hasattr(modelO.model, 'callback_model') and modelO.model.callback_model:
                self.callback_model = modelO.model.callback_model
            else:
                self.callback_model = modelO.model
            self.callbacks.set_model( self.callback_model )
            self.callback_model.stop_training = False
        else:
            self.callbacksS = []
            self.callback_models = []
            filepath = None
            for im,m in enumerate(modelO.models):
                new_cbks = [copy.deepcopy(c) for c in self.cbks]
                for check in new_cbks:
                    if isinstance(check, cbks.ModelCheckpoint):
                        if filepath is None:
                            filepath = check.filepath
                        check.filepath = 'm%d_%s'%(im, filepath)
                        print ("changing file path",check.filepath)                
                #self.callbacksS.append( cbks.CallbackList( self.cbks + [m.history] ))
                self.callbacksS.append( cbks.CallbackList( new_cbks + [m.history] ))
                if hasattr(m, 'callback_model') and m.callback_model:
                    self.callback_models.append( m.callback_model )
                else:
                    self.callback_models.append( m )                    
                self.callbacksS[-1].set_model( self.callback_models[-1] )
                self.callback_models[-1].stop_training = False
                

    def on_train_begin(self):
        if self.callbacks:
            self.callbacks.on_train_begin()
        else:
            for cb in self.callbacksS:
                cb.on_train_begin()

    def on_epoch_begin(self, e):
        if self.callbacks:
            self.callbacks.on_epoch_begin(e)
        else:
            for cb in self.callbacksS:
                cb.on_epoch_begin(e)

    def on_batch_begin(self, i):
        if self.callbacks:
            self.callbacks.on_batch_begin(i)
        else:
            for cb in self.callbacksS:
                cb.on_batch_begin(i)

    def on_batch_end(self, i, l):
        if self.callbacks:
            self.callbacks.on_batch_end(i, l)
        else:
            for cb,lo in zip(self.callbacksS,l):
                cb.on_batch_end(i, lo)
                
    def get_logs(self, metrics, val=False):
        if self.callbacks:
            if val:
                return { 'val_'+name:np.asscalar(metric) for name, metric in
                         zip( self.callback_model.metrics_names, metrics ) }
            else:
                return { name:np.asscalar(metric) for name, metric in
                         zip( self.callback_model.metrics_names, metrics ) }
        else:
            logs = []
            for im,m in enumerate(self.callback_models):
                ametrics = metrics[im,...]
                if val:
                    logs.append({ 'val_'+name:np.asscalar(metric) for name, metric in
                             zip(m.metrics_names, ametrics ) })
                else:
                    logs.append({ name:np.asscalar(metric) for name, metric in
                                  zip(m.metrics_names, ametrics ) })
            return logs

    def on_epoch_end(self, e, logs = None, metrics = None):
        if logs == None:
            logs = self.get_logs( metrics)
        if self.callbacks:
            self.callbacks.on_epoch_end(e, logs )
        else:
            for cb,l in zip(self.callbacksS, logs):
                cb.on_epoch_end(e, l)
                
    def on_train_end(self):
        if self.callbacks:
            self.callbacks.on_train_end()
        else:
            for cb in self.callbacksS:
                cb.on_train_end()

    def stop_training(self):
        if self.callback_model:
            return self.callback_model.stop_training
        else:
            return any([cbm.stop_training for cbm in self.callback_models])
                
class MPIModel(object):
    """Class that abstract all details of the model
    """
    def __init__(self, model=None, models=None):
        self.model = model
        self.models = models
        if model and models:
            raise Exception("Cannot specify single and multiple models")

    def print_metrics(self, metrics):
        if self.model:
            names = self.model.metrics_names
            for name, metric in zip( names, metrics ):
                print ("{0}: {1:.3f}".format(name,metric))
            print ("")
        else:
            for im,m in enumerate(self.models):
                names = m.metrics_names
                ametric = metrics[im,...]
                print ('model {0} {1}'.format( im ,m.name))
                for name, metric in zip( names,ametric):
                    print ("{0}: {1:.3f}".format(name,metric))
                print ("")
                
    def get_logs(self, metrics, val=False):
        if self.model:
            if val:
                return { 'val_'+name:np.asscalar(metric) for name, metric in
                         zip( self.model.metrics_names, metrics ) }
            else:
                return { name:np.asscalar(metric) for name, metric in
                         zip( self.model.metrics_names, metrics ) }
        else:
            logs = []
            for im,m in enumerate(self.models):
                ametrics = metrics[im,...]
                if val:
                    logs.append({ 'val_'+name:np.asscalar(metric) for name, metric in
                             zip(m.metrics_names, ametrics ) })
                else:
                    logs.append({ name:np.asscalar(metric) for name, metric in
                                  zip(m.metrics_names, ametrics ) })
            return logs

    def format_update(self):
        if self.model:
            return [ np.zeros( w.shape, dtype=np.float32 ) for w in self.model.get_weights() ]
        else:
            up = []
            for m in self.models:
                up.append( [ np.zeros( w.shape, dtype=np.float32 ) for w in m.get_weights() ] )
            return up
            
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
            return self.model.history.history
        else:
            return [m.history.history for m in self.models]
            
    def set_history(self, h):
        if self.model:
            self.model.history = h()
        else:
            for m in self.models:
                m.history = h()

    def compile(self, **args):
        if self.model:
            self.model.compile( **args )
        else:
            for m in self.models:
                ## this does not work
                c_args = copy.deepcopy( args )
                m.compile( **c_args )

    #def metrics_names(self):
    #    if self.model:
    #        return self.model.metrics_names
    #    else:
    #        print ("metrics_names not impletment for multi models")
    #        sys.exit(123)
            
    #def callback_model(self):
    #    if self.model:
    #        return getattr(self.model, 'callback_model', None)
        
    def train_on_batch(self, **args):
        if self.model:
            return np.asarray(self.model.train_on_batch( **args ))
        else:
            h = []
            for m in self.models:
                h.append(m.train_on_batch( **args ))
            return np.asarray(h)
                
    def test_on_batch(self, **args):
        if self.model:
            return np.asarray(self.model.test_on_batch( **args ))
        else:
            h= []
            for m in self.models:
                h.append(m.test_on_batch( **args ))
            return np.asarray(h)

    def figure_of_merit(self, **args):
        ## runs like predict trace, and provides a non differentiable figure of merit for hyper-opt
        ## can of course be the validation loss
        return 0.


    def save(self, *args,**kwargs):
        if self.model:
            self.model.save( *args, **kwargs )
        else:
            for im,m in enumerate(self.models):
                fn = 'm%d_%s'%( im, args[0])
                print (fn)
                m.save( fn, **kwargs )

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
                print ("GPU number could not be parsed from {}; using CPU".format(device))
                dev_num = 0
                dev_type = 'cpu'
        else:
            print ("Please specify 'cpu' or 'gpuN' for device name")
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
                self.weights = self.weights.split(',') if self.weights else [None]*len(self.filename)
                for fn,w in zip(self.filename, self.weights):
                    models.append(load_model(filename=fn, weights_file=w))
                return MPIModel(models = models)
            else:
                model = load_model(filename=self.filename, json_str=self.json_str, 
                                   custom_objects=self.custom_objects, weights_file=self.weights)
                return MPIModel(model = model)
