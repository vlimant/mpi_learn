# mpi_learn
Distributed learning with mpi4py

Dependencies: MPI and mpi4py, numpy, keras

Test with the MNIST dataset:
```
python BuildModel.py mnist
python test/get_mnist.py
mpirun -np 3 ./MPIDriver.py mnist train_mnist.list test_mnist.list --loss categorical_crossentropy --epochs 3
```

### Using MPIDriver.py to train your model

`MPIDriver.py` will load a keras model of your choice and train it on the input data you provide.  The script has three required arguments:
- Model name.  By default the script expects a JSON file in the script execution directory called `<model name>_arch.json`.  If you wish to provide an initial set of weights for training, they should be in an HDF5 file called `<model_name>_weights.h5`.  
- File containing a list of training data.  This should be a simple text file with one input data file per line.
- File containing a list of validation data.  This should be a simple text file with one input data file per line.  

See `MPIDriver.py` for supported optional arguments.  Run the script via `mpirun` or `mpiexec`.  It should automatically detect available NVIDIA GPUs and allocate them among the MPI worker processes.
