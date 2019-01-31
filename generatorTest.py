import os, h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, Normalize
plt.switch_backend('Agg')

from EcalEnergyGan import generator as build_generator


#gen_weights='/nfshome/svalleco/mpigan/m0_bs_train_mpi_learn_result.h5'
gen_weights='weights/params_generator_epoch_029.hdf5'
latent_space=200
n_events=100
batch_size=100
np.random.seed()
g = build_generator(latent_space, return_intermediate=False)
g.load_weights(gen_weights)

noise = np.random.normal(0, 1, (n_events, latent_space))
sampled_energies = np.random.uniform(0.1, 5,( batch_size,1 ))
generator_in = np.multiply(sampled_energies, noise)
generated_images = g.predict(generator_in)

generated_images = generated_images.squeeze()
print(generated_images.shape)

fig = plt.figure(1)
plt.imshow(generated_images[10, 12, :, :])
fig.show() 
fig.savefig('event1_yz_goodEx.png')
fig1 = plt.figure(2)
plt.imshow(generated_images[10, :, :, 12])
fig1.show()
fig1.savefig('event1_xy_goodEx.png')
