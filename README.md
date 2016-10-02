# mpi_learn
Distributed learning with mpi4py

Dependencies: MPI and mpi4py, numpy, keras

Test with the MNIST dataset:
```
python BuildModel.py mnist
python test/get_mnist.py
mpirun -np 3 ./MPIDriver.py mnist train_mnist.list test_mnist.list --loss categorical_crossentropy --epochs 3
```
