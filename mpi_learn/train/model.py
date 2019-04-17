### ModelBuilder class and associated helper methods

from mpi_learn.utils import load_model, get_device_name
from .optimizer import OptimizerBuilder
import numpy as np
import copy
import sys
import six
import logging

def session(f):
    def wrapper(*args, **kwargs):
        self_obj = args[0]
        if hasattr(self_obj, 'session'):
            with self_obj.graph.as_default():
                with self_obj.session.as_default():
                    return f(*args, **kwargs)
        else:
            return f(*args, **kwargs)
    return wrapper

class MPIModel(object):
    """Class that abstract all details of the model
    """
    def __init__(self, model=None, models=None):
        self.model = model
        self.models = models
        self.histories = {}
        if model and models:
            raise Exception("Cannot specify single and multiple models")

    @session
    def print_metrics(self, metrics):
        if self.model:
            names = self.model.metrics_names
            for name, metric in zip( names, metrics ):
                logging.info("{0}: {1:.3f}".format(name,metric))
        else:
            for im,m in enumerate(self.models):
                names = m.metrics_names
                ametric = metrics[im,...]
                logging.info('model {0} {1}'.format( im ,m.name))
                for name, metric in zip( names,ametric):
                    logging.info("{0}: {1:.3f}".format(name,metric))
                
    @session
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
        
    @session
    def update_history(self, items, arg_hist):
        if self.model:
            for m,v in items.items():
                arg_hist.setdefault(m,[]).append(v)
        else:
            for im,(m,it) in enumerate(zip(self.models, items)):
                m_name = "model%s"%im
                try:
                    m_name = m.name
                except:
                    logging.warning("no name attr")
                for m,v in it.items():
                    arg_hist.setdefault(m_name,{}).setdefault(m,[]).append(v)
        self.histories = arg_hist
                       
    @session
    def format_update(self):
        if self.model:
            return [ np.zeros( w.shape, dtype=np.float32 ) for w in self.model.get_weights() ]
        else:
            up = []
            for m in self.models:
                up.append( [ np.zeros( w.shape, dtype=np.float32 ) for w in m.get_weights() ] )
            return up
    
    @session
    def get_weights(self):
        if self.model:
            return self.model.get_weights()
        else:
            l_weights = []
            for m in self.models:
                l_weights.append( m.get_weights() )
            return l_weights
    
    @session
    def set_weights(self, w ):
        if self.model:
            self.model.set_weights( w )
        else:
            for m,mw in zip(self.models, w ):
                m.set_weights( mw )
            
    #def history(self):
    #    if self.model:
    #        return self.model.history.history
    #    else:
    #        return [m.history.history for m in self.models]
            
    #def set_history(self, h):
    #    if self.model:
    #        self.model.history = h()
    #    else:
    #        for m in self.models:
    #            m.history = h()

    @session
    def compile(self, **args):
        if 'optimizer' in args and isinstance(args['optimizer'], OptimizerBuilder):
            opt_builder = args['optimizer']
        else:
            opt_builder = None
        if self.model:
            if opt_builder:
                args['optimizer'] = opt_builder.build()
            self.model.compile( **args )
        else:
            for m in self.models:
                if opt_builder:
                    args['optimizer'] = opt_builder.build()
                m.compile( **args )

    @session
    def train_on_batch(self, **args):
        if self.model:
            return np.asarray(self.model.train_on_batch( **args ))
        else:
            h = []
            for m in self.models:
                h.append(m.train_on_batch( **args ))
            return np.asarray(h)
                
    @session
    def test_on_batch(self, **args):
        if self.model:
            return np.asarray(self.model.test_on_batch( **args ))
        else:
            h= []
            for m in self.models:
                h.append(m.test_on_batch( **args ))
            return np.asarray(h)

    @session
    def figure_of_merit(self, **args):
        ## runs like predict trace, and provides a non differentiable figure of merit for hyper-opt
        ## can of course be the validation loss
        if self.model:
            ## return a default value from the validation history
            return (1.-self.histories['val_acc'][-1])
            #return self.histories['val_loss'][-1]
        else: 
            return 0.


    @session
    def save(self, *args,**kwargs):
        if self.model:
            self.model.save( *args, **kwargs )
        else:
            for im,m in enumerate(self.models):
                fn = 'm%d_%s'%( im, args[0])
                m.save( fn, **kwargs )
            
    def close(self):
        if hasattr(self, 'session'):
            self.session.close()
                
class MPITModel(MPIModel):
    """
    adapter of a torch model to fit in the mpi_learn interface
    """
    def __init__(self, model=None, gpus=0):
        MPIModel.__init__(self,model = model)
        self.gpus = gpus
        self.metrics_names = ["loss"]
        self.loss = None
        self.optimizer = None
        self.metrics = []
        self.loss_functions = None

        if self.gpus>0:
            self.model = self.model.cuda()
        if self.gpus >1:
            import torch.nn as nn
            self.model = nn.DataParallel(self.model)
        setattr(self.model, 'metrics_names', self.metrics_names)
        
    def format_update(self):
        ws = self.get_weights()
        return [ np.zeros( w.shape, dtype=np.float32 ) for w in ws]
    
    def get_weights(self):
        if self.gpus > 0:
            return copy.deepcopy([i.data.cpu().numpy() for i in list(self.model.parameters())])
        else:
            return copy.deepcopy([i.data.numpy() for i in list(self.model.parameters())])

    def set_weights(self, weights=[]):
        import torch # Don't put it outside because it will break Tensorflow
        for i,weight in enumerate(weights):
            list(self.model.parameters())[i].data.copy_(torch.from_numpy(weight))

    def compile(self, **kwargs):
        import torch.nn
        import torch.optim

        ### need to map the loss string into the relevant torch object
        self.loss = self._build_loss(kwargs['loss'])
        for metric in kwargs['metrics']:
            if metric.lower() == 'acc' or metric.lower() == 'accuracy':
                self.metrics_names.append('acc')
        ## we need a mapping of the kwargs into what is the optimizer that is getting used
        opt_builder = kwargs.get('optimizer')
        if opt_builder is None:
            self.optimizer = torch.optim.SGD(self.model.parameters(), 1.)
        else:
            self.optimizer = opt_builder.build_torch(self.model)

    def _build_loss(self, loss):
        import torch.nn
        if loss and loss.endswith('Loss'):
            loss = loss[:-4]
        lookup = {
            # Keras mapped losses
            'categorical_crossentropy': torch.nn.CrossEntropyLoss,
            'mean_squared_error' : torch.nn.MSELoss,
            'mean_absolute_error' : torch.nn.L1Loss,
            # PyTorch losses 
            'L1': torch.nn.L1Loss,
            'MSE': torch.nn.MSELoss,
            'CrossEntropy': torch.nn.CrossEntropyLoss,
            #'CTC': torch.nn.CTCLoss,
            'NLL': torch.nn.NLLLoss,
            'PoissonNLL': torch.nn.PoissonNLLLoss,
            'KLDiv': torch.nn.KLDivLoss,
            'BCE': torch.nn.BCELoss,
            'BCEWithLogits': torch.nn.BCEWithLogitsLoss,
            'MarginRanking': torch.nn.MarginRankingLoss,
            'HingeEmbedding': torch.nn.HingeEmbeddingLoss,
            'MultiLabelMargin': torch.nn.MultiLabelMarginLoss,
            'SmoothL1': torch.nn.SmoothL1Loss,
            'SoftMargin': torch.nn.SoftMarginLoss,
            'MultiLabelSoftMargin': torch.nn.MultiLabelSoftMarginLoss,
            'CosineEmbedding': torch.nn.CosineEmbeddingLoss,
            'MultiMargin': torch.nn.MultiMarginLoss,
            'TripletMargin': torch.nn.TripletMarginLoss
        }
        if loss in lookup:
            return lookup[loss]()
        else:
            logger.warning("No loss mapping found, using CrossEntropyLoss")
            return torch.nn.CrossEntropyLoss()

    def _accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1. / batch_size))
        return res
    def _convert_to_tensor(self, data ):
        import torch
        return torch.from_numpy(data)
        #if hasattr(data, 'keys'):
        #    out = [torch.from_numpy(data[key]) for key in sorted(data.keys())]
        #else:
        #    out = torch.from_numpy(data)
        #return out
        
    def train_on_batch(self, x=None, y=None, *args, **kwargs):
        '''Perform a single gradient update on a single batch of data.
        Attributes:
        x: Pytorch tensor of training data
        y: Pytorch tensor of target data

        Return:
        A list of scalar training loss and a metric specified in the compile method.
        '''
        from torch.autograd import Variable
        x = self._convert_to_tensor(x)
        y = self._convert_to_tensor(y)
        self.model.train()
        self.optimizer.zero_grad()
        target = y.long().max(1)[1] # Pytorch doesn't need 1-hot encoded label. Only the indices of classes.
        #target = y.long()
        if self.gpus>0:
            x = x.cuda()
            target = target.cuda()
        x = Variable(x)
        pred = self.model.forward(x)
        loss = self.loss(pred, Variable(target))
        loss.backward()
        self.optimizer.step()
        l_data = loss.data.numpy() if self.gpus == 0 else loss.data.cpu().numpy()
        self.metrics = [l_data] if l_data.shape==() else [l_data[0]]        
        if 'acc' in self.metrics_names: # compute the accuracy
            acc = self._accuracy(pred.data, target, topk=(1,))[0]
            if self.gpus > 0: acc = acc.cpu()
            self.metrics.append(acc.numpy()[0])
        return np.asarray(self.metrics)


    def test_on_batch(self, x=None, y=None, *args, **kwargs):
        '''Test the model on a single batch of samples. No gradient update is performed.
        Attributes:
        x: Pytorch tensor of test data
        y: Pytorch tensor of target data

        Return:
        A list of scalar training loss and a metric specified in the compile method.
        '''
        from torch.autograd import Variable
        x = self._convert_to_tensor(x)
        y = self._convert_to_tensor(y)        
        self.model.eval()
        target = y.long().max(1)[1] # Pytorch doesn't need 1-hot encoded label. Only the indices of classes.
        #target =y.long()
        if self.gpus > 0:
            x = x.cuda()
            target = target.cuda()
        pred = self.model.forward(Variable(x, volatile=True))
        loss = self.loss(pred, Variable(target, volatile=True))
        l_data = loss.data.numpy() if self.gpus == 0 else loss.data.cpu().numpy()
        self.metrics = [l_data] if l_data.shape==() else [l_data[0]]        
        if 'acc' in self.metrics_names: # compute the accuracy
            acc = self._accuracy(pred.data, target, topk=(1,))[0]
            if self.gpus > 0: acc = acc.cpu()
            self.metrics.append(acc.numpy()[0])
        return np.asarray(self.metrics)

    def save(self, *args,**kwargs):
        import torch
        weights_filename = args[0]+'_w'
        arch_filename = args[0]
        torch.save( self.model.state_dict(), weights_filename)
        torch.save( self.model, arch_filename)
            
class ModelBuilder(object):
    """Class containing instructions for building neural net models.
        Derived classes should implement the build_model and get_backend_name
        functions.

        Attributes:
            comm: MPI communicator containing all running MPI processes
    """

    def __init__(self, comm):
        """Arguments:
            comm: MPI communicator 
        """
        self.comm = comm

    def get_device_name(self, device):
        """Should return a device name under a desired convention"""
        return device

    def build_model(self):
        """Should return an uncompiled Keras model."""
        raise NotImplementedError

    def get_backend_name(self):
        """Should return the name of backend framework."""
        raise NotImplementedError

class ModelFromJson(ModelBuilder):
    """ModelBuilder class that builds from model architecture specified
        in a JSON file.
        Attributes:
            filename: path to JSON file specifying model architecture
    """

    def __init__(self, comm, filename=None, custom_objects={}, weights=None):
        self.filename = filename
        self.weights = weights
        self.custom_objects = custom_objects
        super(ModelFromJson, self).__init__(comm)

    def build_model(self, local_session=True):
        if type(self.filename) == list:
            models = []
            for fn in self.filename:
                models.append(load_model(filename=fn))
            return MPIModel(models = models)
        else:        
            return MPIModel(model=load_model(filename=self.filename, custom_objects=self.custom_objects, weights_file=self.weights))

    def get_backend_name(self):
        return 'keras'

class ModelTensorFlow(ModelBuilder):
    """ModelBuilder class that builds from model architecture specified
        in a JSON file. Uses Tensorflow and builds the model on the 
        specified GPU.
        Attributes:
            source: path to JSON file specifying model architecture
                    or Keras model to be cloned
            device: name of the device to use (ex: "/gpu:2")
    """

    def __init__(self, comm, source, device_name='cpu', 
            custom_objects={}, weights=None):
        if isinstance(source, six.string_types):
            if source.endswith('.py'):
                module = __import__(source.replace('.py',''))
                self.model = module.get_model()
                self.filename = None
            else:
                self.filename = source
                self.model = None
        else:
            self.filename = None
            self.model = source
        self.weights = weights
        self.custom_objects = custom_objects
        self.device = self.get_device_name(device_name)
        super(ModelTensorFlow, self).__init__(comm)

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
                logging.warning("GPU number could not be parsed from {}; using CPU".format(device))
                dev_num = 0
                dev_type = 'cpu'
        else:
            logging.warning("Please specify 'cpu' or 'gpuN' for device name; using CPU")
            dev_num = 0
            dev_type = 'cpu'
        return get_device_name(dev_type, dev_num, backend='tensorflow')

    def build_model_aux(self):
        import keras.backend as K

        with K.tf.device(self.device):
            if type(self.filename) == list:
                models = []
                self.weights = self.weights.split(',') if self.weights else [None]*len(self.filename)
                for fn,w in zip(self.filename, self.weights):
                    models.append(load_model(filename=fn, weights_file=w))
                return MPIModel(models = models)
            else:
                model = load_model(filename=self.filename, model=self.model,
                                custom_objects=self.custom_objects, weights_file=self.weights)
                return MPIModel(model = model)


    def build_model(self, local_session = True):
        import keras.backend as K

        if local_session:
            graph = K.tf.Graph()
            session = K.tf.Session(graph=graph, config=K.tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=False,
                gpu_options=K.tf.GPUOptions(
                        per_process_gpu_memory_fraction=1./self.comm.Get_size()) ) )

            with graph.as_default():
                with session.as_default():
                    import keras.backend as K
                    ret_model = self.build_model_aux()
                    ret_model.session = session
                    ret_model.graph = graph
                    return ret_model
        else:
            K.set_session( K.tf.Session( config=K.tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=False,
                gpu_options=K.tf.GPUOptions(
                    per_process_gpu_memory_fraction=1./self.comm.Get_size()) ) ) )
            return self.build_model_aux()

    def get_backend_name(self):
        return 'tensorflow'

class ModelPytorch(ModelBuilder):
    def __init__(self, comm, source,
                 weights = None,
                 gpus=0):
        super(ModelPytorch,self).__init__(comm)
        if isinstance(source, six.string_types):
            if source.endswith('.py'):
                module = __import__(source.replace('.py',''))
                self.model = module.get_model()
                self.filename = None
            else:
                self.filename = source
                self.model = None
        else:
            self.filename = None
            self.model = source
        self.weights = weights
        self.gpus=gpus

    def build_model(self, local_session=True):
        import torch
        if self.filename is not None:
            model = torch.load(self.filename)
        elif self.model is not None:
            model = copy.deepcopy(self.model)
        if self.weights:
            wd = torch.load(self.weights)
            model.load_state_dict(wd)
        return MPITModel(model=model, gpus=self.gpus)

    def get_backend_name(self):
        return 'pytorch'
                                                        
