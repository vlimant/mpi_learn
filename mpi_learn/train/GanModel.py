#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle
import keras
from keras.models import Model
from keras.layers import Input
from keras import optimizers
from keras.optimizers import RMSprop,SGD
#from EcalEnergyGan import generator, discriminator
import numpy as np
import numpy.core.umath_tests as umath
import time
import socket
import os
import glob
import h5py


import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import (Input, Dense, Reshape, Flatten, Lambda, merge,
                          Dropout, BatchNormalization, Activation, Embedding)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import (UpSampling3D, Conv3D, ZeroPadding3D,
                                        AveragePooling3D)

from mpi_learn.utils import get_device_name
from mpi_learn.train.model import MPIModel, ModelBuilder
from .optimizer import OptimizerBuilder

import keras
kv2 = keras.__version__.startswith('2')

def hn():
    return socket.gethostname()


def weights(m):
    _weights_names = []
    for layer in m.layers:
        _weights_names += [ll.name for ll in layer.weights]
    _weights = m.get_weights()
    _disp = [(np.min(s),np.max(s),np.mean(s),np.std(s),s.shape,n) for s,n in zip(_weights,_weights_names)]
    for ii,dd in enumerate(_disp):
        print (ii,dd)

def weights_diff( m ,lap=True, init=False,label='', alert=None):#1000.):
    if (weights_diff.old_weights is None) or init:
        weights_diff.old_weights = m.get_weights()
        return
    _weights_names = []
    for layer in m.layers:
        _weights_names += [ll.name for ll in layer.weights]

    check_on_weight = m.get_weights()
    and_check_on_weight = weights_diff.old_weights
    ## make the diffs
    _diffs = [np.subtract(a,b) for (a,b) in zip(check_on_weight,and_check_on_weight)]
    _diffsN = [(np.min(s),np.max(s),np.mean(s),np.std(s),s.shape,n) for s,n in zip(_diffs,_weights_names)]
    #print ('\n'.join(['%s'%dd for dd in _diffsN]))
    for ii,dd in enumerate(_diffsN):
        if alert:
            if not any([abs(vv) > alert for vv in dd[:3]]):
                continue
        print (ii,'WD %s'%label,dd)
        #if dd[-2] == (8,):
        #    print ("\t",_diffs[ii])
    if lap:
        weights_diff.old_weights = m.get_weights()

weights_diff.old_weights = None

def _Conv3D(N,a,b,c,**args):
    if kv2:
        if 'border_mode'in args: args['padding'] = args.pop('border_mode')
        if 'init' in args: args['kernel_initializer'] = args.pop('init')
        if 'bias' in args: args['use_bias'] = args.pop('bias')
        return Conv3D(N,(a,b,c), **args)
    else:
        return Conv3D(N, a,b,c, **args)
def _BatchNormalization(**args):
    if kv2:
        m=0
        if 'mode' in args:
            m=args.pop('mode')
        if m==2:
            return StaticBatchNormalization(**args)
            #return BatchNormalization(**args)
        else:
            #args['scale'] = False
            #args['center'] = False
            return BatchNormalization(**args)
    else:
        return BatchNormalization(**args)

def _Dense(N,**args):
    if kv2:
        if 'init' in args: args['kernel_initializer'] = args.pop('init')
        return Dense(N,**args)
    else:
        return Dense(N,**args)

def _Model(**args):
    if kv2:
        args['outputs'] = args.pop('output')
        args['inputs'] = args.pop('input')
        return Model(**args)
    else:
        return Model(**args)
def discriminator(fixed_bn = False, discr_drop_out=0.2):
    if keras.backend.image_data_format() =='channels_last':
        dshape=(25, 25, 25,1)
        daxis=(1,2,3)
    else:
        dshape=(1, 25, 25, 25)
        daxis=(2,3,4)

    image = Input(shape=dshape, name='image')



    bnm=2 if fixed_bn else 0
    f=(5,5,5)
    x = _Conv3D(32, 5, 5,5,border_mode='same',
               name='disc_c1')(image)
    x = LeakyReLU()(x)
    x = Dropout(discr_drop_out)(x)

    x = ZeroPadding3D((2, 2,2))(x)
    x = _Conv3D(8, 5, 5,5,border_mode='valid',
               name='disc_c2'
    )(x)
    x = LeakyReLU()(x)
    x = _BatchNormalization(name='disc_bn1',
                           mode=bnm,
    )(x)
    x = Dropout(discr_drop_out)(x)

    x = ZeroPadding3D((2, 2, 2))(x)
    x = _Conv3D(8, 5, 5,5,border_mode='valid',
               name='disc_c3'
)(x)
    x = LeakyReLU()(x)
    x = _BatchNormalization(name='disc_bn2',
                           #momentum = 0.00001
                           mode=bnm,
    )(x)
    x = Dropout(discr_drop_out)(x)

    x = ZeroPadding3D((1, 1, 1))(x)
    x = _Conv3D(8, 5, 5,5,border_mode='valid',
               name='disc_c4'
    )(x)
    x = LeakyReLU()(x)
    x = _BatchNormalization(name='disc_bn3',
                           mode=bnm,
    )(x)
    x = Dropout(discr_drop_out)(x)

    x = AveragePooling3D((2, 2, 2))(x)
    h = Flatten()(x)

    dnn = _Model(input=image, output=h, name='dnn')

    dnn_out = dnn(image)

    fake = _Dense(1, activation='sigmoid', name='classification')(dnn_out)
    aux = _Dense(1, activation='linear', name='energy')(dnn_out)
    ecal = Lambda(lambda x: K.sum(x, daxis), name='sum_cell')(image)

    return _Model(output=[fake, aux, ecal], input=image, name='discriminator_model')

def generator(latent_size=200, return_intermediate=False, with_bn=True):
    if keras.backend.image_data_format() =='channels_last':
        dim = (7,7,8,8)
    else:
        dim = (8, 7, 7,8)
    latent = Input(shape=(latent_size, ))
    bnm=0
    x = _Dense(64 * 7* 7, init='glorot_normal',
              name='gen_dense1'
    )(latent)
    x = Reshape(dim)(x)
    x = _Conv3D(64, 6, 6, 8, border_mode='same', init='he_uniform',
               name='gen_c1'
    )(x)
    x = LeakyReLU()(x)
    if with_bn:
        x = _BatchNormalization(name='gen_bn1',
                           mode=bnm
    )(x)
    x = UpSampling3D(size=(2, 2, 2))(x)

    x = ZeroPadding3D((2, 2, 0))(x)
    x = _Conv3D(6, 6, 5, 8, init='he_uniform',
               name='gen_c2'
    )(x)
    x = LeakyReLU()(x)
    if with_bn:
        x = _BatchNormalization(name='gen_bn2',
                           mode=bnm)(x)
    x = UpSampling3D(size=(2, 2, 3))(x)

    x = ZeroPadding3D((1,0,3))(x)
    x = _Conv3D(6, 3, 3, 8, init='he_uniform',
               name='gen_c3')(x)
    x = LeakyReLU()(x)

    x = _Conv3D(1, 2, 2, 2, bias=False, init='glorot_normal',
               name='gen_c4')(x)
    x = Activation('relu')(x)

    loc = _Model(input=latent, output=x)
    fake_image = loc(latent)
    _Model(input=[latent], output=fake_image)
    return _Model(input=[latent], output=fake_image, name='generator_model')

def get_sums(images):
    if keras.backend.image_data_format() =='channels_last':
       sumsx = np.squeeze(np.sum(images, axis=(2,3)))
       sumsy = np.squeeze(np.sum(images, axis=(1,3)))
       sumsz = np.squeeze(np.sum(images, axis=(1,2)))
    else: 
       sumsx = np.squeeze(np.sum(images, axis=(3,4)))
       sumsy = np.squeeze(np.sum(images, axis=(2,4)))
       sumsz = np.squeeze(np.sum(images, axis=(2,3)))
    return sumsx, sumsy, sumsz

def get_moments(images, sumsx, sumsy, sumsz, totalE, m):
    ecal_size = 25
    totalE = np.squeeze(totalE)
    index = images.shape[0]
    momentX = np.zeros((index, m))
    momentY = np.zeros((index, m))
    momentZ = np.zeros((index, m))
    ECAL_midX = np.zeros(index)
    ECAL_midY = np.zeros(index)
    ECAL_midZ = np.zeros(index)
    if (totalE==0).any(): return momentX, momentY, momentZ
    for i in range(m):
      relativeIndices = np.tile(np.arange(ecal_size), (index,1))
      moments = np.power((relativeIndices.transpose()-ECAL_midX).transpose(), i+1)
      ECAL_momentX = umath.inner1d(sumsx, moments) /totalE
      if i==0: ECAL_midX = ECAL_momentX.transpose()
      momentX[:,i] = ECAL_momentX
    for i in range(m):
      relativeIndices = np.tile(np.arange(ecal_size), (index,1))
      moments = np.power((relativeIndices.transpose()-ECAL_midY).transpose(), i+1)
      ECAL_momentY = umath.inner1d(sumsy, moments) /totalE
      if i==0: ECAL_midY = ECAL_momentY.transpose()
      momentY[:,i]= ECAL_momentY
    for i in range(m):
      relativeIndices = np.tile(np.arange(ecal_size), (index,1))
      moments = np.power((relativeIndices.transpose()-ECAL_midZ).transpose(), i+1)
      ECAL_momentZ = umath.inner1d(sumsz, moments)/totalE
      if i==0: ECAL_midZ = ECAL_momentZ.transpose()
      momentZ[:,i]= ECAL_momentZ
    return momentX, momentY, momentZ

def load_sorted(sorted_path):
    sorted_files = sorted(glob.glob(sorted_path))

    print ("found sorterd files",sorted( sorted_files))
    energies = []
    srt = {}
    for f in sorted_files:
        print (f)
        #energy = int(list(filter(str.isdigit, f))[:-1])
        file_name=f[f.find('sorted_'):-1]
        #energy = int(''.join(list(filter(str.isdigit, f))[:-1]))
        energy = int(''.join(list(filter(str.isdigit, file_name))[:-1]))*10
        print ("found files for energy",energy)
        energies.append(energy)
        srtfile = h5py.File(f,'r')
        srt["events" + str(energy)] = np.array(srtfile.get('ECAL'))
        srt["ep" + str(energy)] = np.array(srtfile.get('Target'))
    return energies, srt

def generate(g, index, latent, sampled_labels):
    noise = np.random.normal(0, 1, (index, latent))
    sampled_labels=np.expand_dims(sampled_labels, axis=1)
    gen_in = np.multiply(sampled_labels, noise)
    generated_images = g.predict(gen_in, verbose=False, batch_size=100)
    return generated_images



def metric(ganvar, g4var, energies, m):
   #calculate totale absolute errors on average moments+energies 
   ecal_size = 25
   metricp = 0
   metrice = 0
   for energy in energies:
     #Relative error on mean moment value for each moment and each axis
     if (g4var['moms_x'+str(energy)].all() > 0.0) :
        posx_error = (g4var['moms_x'+str(energy)] - ganvar['moms_x'+str(energy)])/g4var['moms_x'+str(energy)]
     else:
        posx_error = 1.
     if (g4var['moms_y'+str(energy)].all() > 0.0) :
        posy_error = (g4var['moms_y'+str(energy)] - ganvar['moms_y'+str(energy)])/g4var['moms_y'+str(energy)]
     else:
        posy_error = 1.
     if (g4var['moms_z'+str(energy)].all() > 0.0) :
        posz_error = (g4var['moms_z'+str(energy)] - ganvar['moms_z'+str(energy)])/g4var['moms_z'+str(energy)]
     else:
        posz_error = 1.
     #Taking absolute of errors and adding for each axis then scaling by 3
     pos_error = (np.absolute(posx_error) + np.absolute(posy_error) + np.absolute(posz_error))/3
     #Summing over moments and dividing for number of moments
     metricp += np.sum(pos_error)/m
     #Take profile along each axis and find mean along events
     eprofilex_error=0
     eprofiley_error=0
     eprofilez_error=0
     if (g4var['sumx'+str(energy)].all() > 0.0) :
        eprofilex_error = np.divide((g4var['sumx'+str(energy)] - ganvar['sumx'+str(energy)]), g4var['sumx'+str(energy)])
     else:
        eprofilex_error  =1.
        #raise ValueError('Image with NULL energy!')
     if (g4var['sumy'+str(energy)].all() > 0.0) :
        eprofiley_error = np.divide((g4var['sumy'+str(energy)] - ganvar['sumy'+str(energy)]), g4var['sumy'+str(energy)])
     else:
        eprofiley_error  =1.
        #raise ValueError('Image with NULL energy!')
     if (g4var['sumy'+str(energy)].all() > 0.0) :
        eprofilez_error = np.divide((g4var['sumz'+str(energy)] - ganvar['sumz'+str(energy)]), g4var['sumz'+str(energy)])
     else:
        eprofilez_error  =1.
        #raise ValueError('Image with NULL energy!')
     #Take absolute of error and mean for all events
     eprofilex_total = np.sum(np.absolute(eprofilex_error))/ecal_size
     eprofiley_total = np.sum(np.absolute(eprofiley_error))/ecal_size
     eprofilez_total = np.sum(np.absolute(eprofilez_error))/ecal_size

     metrice +=(eprofilex_total + eprofiley_total + eprofilez_total)/3
   tot = (metricp + metrice)/len(energies)
   return(tot)



def bit_flip(x, prob=0.05):
    """ flips a int array's values with some probability """
    x = np.array(x)
    selection = np.random.uniform(0, 1, x.shape) < prob
    x[selection] = 1 * np.logical_not(x[selection])
    return x


class StaticBatchNormalization(BatchNormalization):
    def call(self, inputs, training=None):
        return super(StaticBatchNormalization, self).call(inputs, training=False)

class GANModel(MPIModel):
    def __init__(self, **args):#latent_size=200, checkpoint=True, gen_bn=True, onepass=False):
        self.tell = args.get('tell',True)
        self.gen_bn = args.get('gen_bn',True)
        self._onepass = args.get('onepass',bool(int(os.environ.get('GANONEPASS',0))))
        self._reversedorder = args.get('reversedorder',bool(int(os.environ.get('GANREVERSED',0))))
        self._switchingloss = args.get('switchingloss',False)
        self._heavycheck = args.get('heavycheck',False)
        self._show_values = args.get('show_values',False)
        self._show_loss = args.get('show_loss', False)
        self._show_weights = False

        self.latent_size=args.get('latent_size',200)
        self.discr_drop_out=args.get('discr_drop_out',0.2)
        self.batch_size= None ## will be taken from the data that is passed on
        self.discr_loss_weights = [
            args.get('gen_weight',2),
            args.get('aux_weight',0.1),
            args.get('ecal_weight',0.1)
        ]
        
        self.with_fixed_disc = args.get('with_fixed_disc',True)
        self.assemble_models()
        self.recompiled = False
        self.checkpoint = args.get('checkpoint',int(os.environ.get('GANCHECKPOINT',0)))
        self.calculate_fom = args.get('calculate_fom',True)

        if self.tell:
            print ("Generator summary")
            self.generator.summary()
            print ("Discriminator summary")
            self.discriminator.summary()
            print ("Combined summary")
            self.combined.summary()
        if True:
            if self.with_fixed_disc: print ("the batch norm weights are fixed. heavey weight re-assigning")
            if self.checkpoint: print ("Checkpointing the model weigths after %d batch, based on the process id"%self.checkpoint)
            if self._onepass: print ("Training in one pass")
            if self._reversedorder: print ("will train generator first, then discriminator")
            if self._heavycheck: print("running heavy check on weight sanity")
            if self._show_values: print("showing the input values at each batch")
            if self._show_loss: print("showing the loss at each batch")
            if self._show_weights: print("showing weights statistics at each batch")

        MPIModel.__init__(self, models = [
            self.discriminator,
            #self.generator
            self.combined ## this is increasing a bit the amount of com by sending twice the discriminator
        ])

        
        ## counters
        self.g_cc = 0
        self.d_cc = 0
        self.p_cc = 0
        self.g_t = []
        self.d_t = []
        self.p_t = []


    def big_assemble_models(self):

        image = Input(shape=( 25, 25, 25,1 ), name='image')

        x = _Conv3D(32, 5, 5,5,border_mode='same')(image)
        x = LeakyReLU()(x)
        x = Dropout(discr_drop_out)(x)

        x = ZeroPadding3D((2, 2,2))(x)
        x = _Conv3D(8, 5, 5,5,border_mode='valid')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(discr_drop_out)(x)

        x = ZeroPadding3D((2, 2, 2))(x)
        x = Conv3D(8, 5, 5,5,border_mode='valid')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(discr_drop_out)(x)

        x = ZeroPadding3D((1, 1, 1))(x)
        x = Conv3D(8, 5, 5,5,border_mode='valid')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(discr_drop_out)(x)

        x = AveragePooling3D((2, 2, 2))(x)
        h = Flatten()(x)


        fake = Dense(1, activation='sigmoid', name='generation')(h)
        aux = Dense(1, activation='linear', name='auxiliary')(h)
        ecal = Lambda(lambda x: K.sum(x, axis=(1, 2, 3)), name='sum_cell')(image)

        self.discriminator = Model(output=[fake, aux, ecal], input=image, name='discriminator_model')

        latent = Input(shape=(self.latent_size, ))

        x = Dense(64 * 7* 7, init='he_uniform')(latent)
        x = Reshape((7, 7,8, 8))(x)
        x = Conv3D(64, 6, 6, 8, border_mode='same', init='he_uniform' )(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = UpSampling3D(size=(2, 2, 2))(x)

        x = ZeroPadding3D((2, 2, 0))(x)
        x = Conv3D(6, 6, 5, 8, init='he_uniform')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = UpSampling3D(size=(2, 2, 3))(x)

        x = ZeroPadding3D((1,0,3))(x)
        x = Conv3D(6, 3, 3, 8, init='he_uniform')(x)
        x = LeakyReLU()(x)

        x = Conv3D(1, 2, 2, 2, bias=False, init='glorot_normal')(x)
        x = Activation('relu')(x)

        loc = Model(latent, x)
        #fake_image = loc(latent)
        self.generator = Model(input=latent, output=x, name='generator_model')

        c_fake, c_aux, c_ecal = self.discriminator(x)
        self.combined = Model(
            input = latent,
            output = [c_fake, c_aux, c_ecal],
            name='combined_model'
            )


    def ext_assemble_models(self):
        print('[INFO] Building generator')
        self.generator = generator(self.latent_size, with_bn = self.gen_bn)
        print('[INFO] Building discriminator')
        self.discriminator = discriminator(discr_drop_out = self.discr_drop_out)
        if self.with_fixed_disc:
            self.fixed_discriminator = discriminator(discr_drop_out = self.discr_drop_out, fixed_bn=True)
        print('[INFO] Building combined')
        latent = Input(shape=(self.latent_size, ), name='combined_z')
        fake_image = self.generator(latent)
        if self.with_fixed_disc:
            fake, aux, ecal = self.fixed_discriminator(fake_image)
        else:
            fake, aux, ecal = self.discriminator(fake_image)

        self.combined = Model(
            input=[latent],
            output=[fake, aux, ecal],
            name='combined_model'
        )

    def compile(self, **args):
        ## args are fully ignored here
        print('[INFO] IN GAN MODEL: COMPILE')
        if 'optimizer' in args and isinstance(args['optimizer'], OptimizerBuilder):
            opt_builder = args['optimizer']
        else:
            opt_builder = None

        def make_opt(**args):
            if opt_builder:
                opt = opt_builder.build()
            else:
                ## there are things specified from outside mpi-learn
                lr = args.get('lr',0.0001)
                prop = args.get('prop',True) ## gets as the default
                if prop:
                    opt = RMSprop()    
                else:
                    opt = SGD(lr=lr)

            print ("optimizer for compiling",opt) 
            return opt

        self.generator.compile(
            optimizer=make_opt(**args),
            loss='binary_crossentropy') ## never  actually used for training

        self.discriminator.compile(
            optimizer=make_opt(**args),
            loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
            loss_weights=self.discr_loss_weights
        )

        if hasattr(self,'fixed_discriminator'):
            self.fixed_discriminator.trainable = False
        else:
            self.discriminator.trainable = False

        self.combined.compile(
            optimizer=make_opt(**args),
            loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
            loss_weights=self.discr_loss_weights
        )
        if kv2: 
           self.discriminator.trainable = True #workaround for keras 2 bug

        self.combined.metrics_names = self.discriminator.metrics_names
        print (self.discriminator.metrics_names)
        print (self.combined.metrics_names)

        
        if hasattr(self, 'calculate_fom'):
            self.energies, self.g4var = self.prepare_geant4_data()
         
        print ("compiled")

    def assemble_models(self):
        self.ext_assemble_models()

    def batch_transform(self, x, y):
        root_fit = [0.0018, -0.023, 0.11, -0.28, 2.21]
        x_disc_real =x
        y_disc_real =y
        show_values = self._show_values
        def mm( label, t):
            print (label,np.min(t),np.max(t),np.mean(t),np.std(t),t.shape)

        if self.batch_size is None:
            ## fix me, maybe
            self.batch_size = x_disc_real.shape[0]
            print (hn(),"initializing sizes",x_disc_real.shape,[ yy.shape for yy in y])


        noise = np.random.normal(0, 1, (self.batch_size, self.latent_size))
        sampled_energies = np.random.uniform(0.1, 5,(self.batch_size,1))
        generator_ip = np.multiply(sampled_energies, noise)
        #if show_values: print ('energies',np.ravel(sampled_energies)[:10])
        if show_values: mm('energies',sampled_energies)
        ratio = np.polyval(root_fit, sampled_energies)
        #if show_values: print ('ratios',np.ravel(ratio)[:10])
        if show_values: mm('ratios',ratio)
        ecal_ip = np.multiply(ratio, sampled_energies)
        #if show_values: print ('estimated sum cells',np.ravel(ecal_ip)[:10])
        if show_values: mm('estimated sum cells',ecal_ip)

        now = time.mktime(time.gmtime())
        if self.p_cc>1 and len(self.p_t)%100==0:
            print ("prediction average",np.mean(self.p_t),"[s]' over",len(self.p_t))
        generated_images = self.generator.predict(generator_ip)
        ecal_rip = np.squeeze(np.sum(generated_images, axis=(1, 2, 3)))
        #if show_values: print ('generated sum cells',np.ravel(ecal_rip)[:10])
        if show_values: mm('generated sum cells',ecal_rip)

        norm_overflow = False
        apply_identify = False ## False was intended originally

        if norm_overflow and np.max( ecal_rip ) > 1000.:
            if show_values: print ("normalizing back")
            #ecal_ip = ecal_rip
            generated_images /= np.max( generated_images )
            ecal_rip = np.squeeze(np.sum(generated_images, axis=(1, 2, 3)))
            #if show_values: print ('generated sum cells',np.ravel(ecal_rip)[:10])
            if show_values: mm('generated sum cells',ecal_rip)
        elif apply_identify:
            ecal_ip = ecal_rip

        done = time.mktime(time.gmtime())
        if self.p_cc:
            self.p_t.append( done - now )
        self.p_cc +=1

        ## need to bit flip the true labels too
        bf = bit_flip(y[0])
        y_disc_fake = [bit_flip(np.zeros(self.batch_size)), sampled_energies.reshape((-1,)), ecal_ip.reshape((-1,))]
        re_y = [bf, y_disc_real[1], y_disc_real[2]]

        if self._onepass:
            ## train the discriminator in one go
            bb_x  = np.concatenate( (x_disc_real, generated_images))
            bb_y = [np.concatenate((a,b)) for a,b in zip(re_y, y_disc_fake)]
            rng_state = np.random.get_state()
            np.random.shuffle( bb_x )
            np.random.set_state(rng_state)
            np.random.shuffle( bb_y[0] )
            np.random.set_state(rng_state)
            np.random.shuffle( bb_y[1] )
            np.random.set_state(rng_state)
            np.random.shuffle( bb_y[2] )

            X_for_disc = bb_x
            Y_for_disc = bb_y



        c_noise = np.random.normal(0, 1, (2*self.batch_size, self.latent_size))
        ###print ('noise',np.ravel(noise)[:10])
        c_sampled_energies = np.random.uniform(0.1, 5, (2*self.batch_size,1 ))
        c_generator_ip = np.multiply(c_sampled_energies, c_noise)
        c_ratio = np.polyval(root_fit, c_sampled_energies)
        c_ecal_ip = np.multiply(c_ratio, c_sampled_energies)
        c_trick = np.ones(2*self.batch_size)

        X_for_combined = [c_generator_ip]
        Y_for_combined = [c_trick,c_sampled_energies.reshape((-1, 1)), c_ecal_ip]

        if self._onepass:
            return (X_for_disc,Y_for_disc,X_for_combined,Y_for_combined)
        else:
            def head(a):
                return [o[:self.batch_size] for o in a]
            def tail(a):
                return [o[self.batch_size:] for o in a]

            return ((x_disc_real,re_y),(generated_images,y_disc_fake), (head(X_for_combined),head(Y_for_combined)), (tail(X_for_combined),tail(Y_for_combined)))

    def test_on_batch(self,x, y, sample_weight=None):
        show_loss = self._show_loss
        if self._onepass:
            (X_for_disc,Y_for_disc,X_for_combined,Y_for_combined) = self.batch_transform(x,y)
            epoch_disc_loss = self.discriminator.test_on_batch(X_for_disc,Y_for_disc)
            epoch_gen_loss = self.combined.test_on_batch(X_for_combined,Y_for_combined)
            if show_loss:
                print ("test discr loss",epoch_disc_loss)
                print ("test combined loss",epoch_gen_loss)
        else:
            ((x_disc_real,re_y),(generated_images, y_disc_fake),(x_comb1,y_comb1),(x_comb2,y_comb2)) = self.batch_transform(x,y)
            real_disc_loss = self.discriminator.test_on_batch( x_disc_real,re_y )
            fake_disc_loss = self.discriminator.test_on_batch( generated_images, y_disc_fake)
            epoch_disc_loss = [(a + b) / 2 for a, b in zip(real_disc_loss, fake_disc_loss)]

            c_loss1= self.combined.test_on_batch( x_comb1,y_comb1 )
            c_loss2= self.combined.test_on_batch(x_comb2,y_comb2 )
            epoch_gen_loss = [(a + b) / 2 for a, b in zip(c_loss1,c_loss2)]
            if show_loss:
                print ("test discr loss",real_disc_loss,fake_disc_loss)
                print ("test combined loss",c_loss1, c_loss2)




        return np.asarray([epoch_disc_loss, epoch_gen_loss])

    def train_on_batch(self, x, y,
                   sample_weight=None,
                   class_weight=None):

        with np.errstate( divide='raise', invalid='raise' , over='raise', under ='raise' ):
            if self._onepass:
                return self._onepass_train_on_batch(x,y,sample_weight,class_weight)
            else:
                return self._twopass_train_on_batch(x,y,sample_weight,class_weight)
    def _checkpoint(self):
        if self.checkpoint and (self.g_cc%self.checkpoint)==0:
            dest='%s/mpi_generator_%s_%s.h5'%(os.environ.get('GANCHECKPOINTLOC','.'),socket.gethostname(),os.getpid())
            print ("Saving generator to",dest,"at",self.g_cc)
            self.generator.save_weights(dest)        

    def _onepass_train_on_batch(self, x, y,
                   sample_weight=None,
                   class_weight=None):

        show_weights = self._show_weights
        show_loss = self._show_loss
        (X_for_disc,Y_for_disc,X_for_combined,Y_for_combined) = self.batch_transform(x,y)


        if self._heavycheck:
            on_weight = self.combined
            check_on_weight = on_weight.get_weights()
            if self._show_weights:
                weights( on_weight )
            weights_diff( on_weight , init=True)


        def _train_disc():
            self.discriminator.trainable = True
            now = time.mktime(time.gmtime())
            epoch_disc_loss = self.discriminator.train_on_batch(X_for_disc,Y_for_disc)
            if show_loss:
                print (self.d_cc," discr loss",epoch_disc_loss)
            done = time.mktime(time.gmtime())
            if self.d_cc:
                self.d_t.append( done - now )
            self.d_cc+=1
            if hasattr(self,'fixed_discriminator'):
                self.fixed_discriminator.set_weights( self.discriminator.get_weights())
            return epoch_disc_loss

        def _train_comb(noT=False):
            if hasattr(self,'fixed_discriminator'):
                self.fixed_discriminator.trainable = False
            else:
                self.discriminator.trainable = False
            now = time.mktime(time.gmtime())
            if noT:
                print ("evaluating the combined model")
                epoch_gen_loss = self.combined.test_on_batch(X_for_combined,Y_for_combined)
            else:
                epoch_gen_loss = self.combined.train_on_batch(X_for_combined,Y_for_combined)

            if show_loss:
                print (self.g_cc,"combined loss",epoch_gen_loss)
            done = time.mktime(time.gmtime())
            if self.g_cc:
                self.g_t.append( done - now )
            self.g_cc+=1
            return epoch_gen_loss

        if self._reversedorder:
            epoch_gen_loss = _train_comb(noT=(self.g_cc==0))
            _pass = 'C-pass'
        else:
            epoch_disc_loss = _train_disc()
            _pass = 'D-pass'

        if self._heavycheck:
            if show_weights: weights( on_weight )
            weights_diff( on_weight , label=_pass)

        if self._reversedorder:
            epoch_disc_loss = _train_disc()
            _pass = 'D-pass'
        else:
            epoch_gen_loss = _train_comb()
            _pass = 'C-pass'

        if self._heavycheck:
            if show_weights: weights( on_weight )
            weights_diff( on_weight , label=_pass)

        if show_weights:
            weights( self.discriminator )
            weights( self.generator )
            weights( self.combined )


        if len(self.g_t)>0 and len(self.g_t)%100==0:
            print ("generator average ",np.mean(self.g_t),"[s] over",len(self.g_t))

        if len(self.d_t)>0 and len(self.d_t)%100==0:
            print ("discriminator average",np.mean(self.d_t),"[s] over ",len(self.d_t))

        self._checkpoint()

        return np.asarray([epoch_disc_loss, epoch_gen_loss])

    def _twopass_train_on_batch(self, x, y,
                   sample_weight=None,
                   class_weight=None):

        ((x_disc_real,re_y),(generated_images, y_disc_fake),(x_comb1,y_comb1),(x_comb2,y_comb2)) = self.batch_transform(x,y)

        show_loss = self._show_loss
        show_weights = self._show_weights
        if self.d_cc>1 and len(self.d_t)%100==0:
            print ("discriminator average",np.mean(self.d_t),"[s] over ",len(self.d_t))
        self.discriminator.trainable = True

        if self._heavycheck:
            #on_weight = self.generator
            on_weight = self.combined
            check_on_weight = on_weight.get_weights()
            weights_names = []
            for l in on_weight.layers:
                weights_names += [ll.name for ll in l.weights]
            if self._show_weights:
                weights( on_weight )
            weights_diff( on_weight , init=True)

        now = time.mktime(time.gmtime())
        real_batch_loss = self.discriminator.train_on_batch(x_disc_real,re_y)

        if self._heavycheck:
            if hasattr(self,'fixed_discriminator'):
                self.fixed_discriminator.set_weights( self.discriminator.get_weights())
            if show_weights: weights( on_weight )
            weights_diff( on_weight , label='D-real')

        fake_batch_loss = self.discriminator.train_on_batch(generated_images, y_disc_fake)

        if hasattr(self,'fixed_discriminator'):
            ## pass things over
            self.fixed_discriminator.set_weights( self.discriminator.get_weights())
            self.fixed_discriminator.trainable = False
        else:
            self.discriminator.trainable = False

        if self._heavycheck:
            if show_weights: weights( on_weight )
            weights_diff( on_weight , label='D-fake')


        if show_loss:
            #print (self.discriminator.metrics_names)
            print (self.d_cc,"discr loss",real_batch_loss,fake_batch_loss)
        epoch_disc_loss = np.asarray([(a + b) / 2 for a, b in zip(real_batch_loss, fake_batch_loss)])
        done = time.mktime(time.gmtime())
        if self.d_cc:
            self.d_t.append( done - now )
        self.d_cc+=1

        if show_weights:
            weights( self.discriminator )
            weights( self.generator )
            weights( self.combined )

        if self.g_cc>1 and len(self.g_t)%100==0:
            print ("generator average ",np.mean(self.g_t),"[s] over",len(self.g_t))
            now = time.mktime(time.gmtime())

        if self.g_cc:
            self.g_t.append( done - now )
        c_loss1= self.combined.train_on_batch( x_comb1,y_comb1)
        if self._heavycheck:
            if show_weights: weights( on_weight )
            weights_diff( on_weight , label ='C-1')
        c_loss2= self.combined.train_on_batch( x_comb2,y_comb2)
        #c_loss2= c_loss1
        if self._heavycheck:
            if show_weights: weights( on_weight )
            weights_diff( on_weight , label='C-2')

        if show_loss:
            #print(self.combined.metrics_names)
            print (self.g_cc,"combined loss",c_loss1,c_loss2)
        epoch_gen_loss = np.asarray([(a + b) / 2 for a, b in zip(c_loss1,c_loss2)])
        done = time.mktime(time.gmtime())
        if self.g_cc:
            self.g_t.append( done - now )
        self.g_cc+=1

        if self._heavycheck:
            and_check_on_weight = on_weight.get_weights()

            ## this contains a boolean whether all values of the tensors are equal :
            # [--, False, --] ===> weights have changed for one layer
            # [--, True, --] ===> weights are the same one layer
            if show_weights:
                checks = [np.all(np.equal(a,b)) for (a,b) in zip(check_on_weight,and_check_on_weight)]
                weights_have_changed = not all(checks)
                weights_are_all_equal = all(checks)
                print ('Weights are the same?',checks)
                if weights_have_changed:
                    for iw,b in enumerate(checks):
                        if not b:
                            print (iw,"This",check_on_weight[iw].shape)
                            print (np.ravel(check_on_weight[iw])[:10])
                            print (iw,"And that",and_check_on_weight[iw].shape)
                            print (np.ravel(and_check_on_weight[iw])[:10])
                else:
                    print ("weights are all identical")
                    print (np.ravel(and_check_on_weight[1])[:10])
                    print (np.ravel(check_on_weight[1])[:10])

        self._checkpoint()

        ## modify learning rate
        if self._switchingloss:
            switching_loss = (1.,1.)
            if False and not self.recompiled and epoch_disc_loss[0]<switching_loss[0] and epoch_gen_loss[0]<switching_loss[1]:
                ## go on
                print ("going for full sgd")
                self.recompiled = True
                self.compile( prop=False, lr=1.0)
                #K.set_value( self.discriminator.optimizer.lr, 1.0)
                #K.set_value( self.generator.optimizer.lr, 1.0)

            lr = K.get_value(self.discriminator.optimizer.lr)
            nlr = lr
            ths = [
                #(5.5, 1.0),
                #(10., 0.1),
                (1000., 0.01),
                (10000., 0.001) ,
            ]
            for th in sorted(ths, key = lambda o : o[0]):
                if epoch_disc_loss[0] < th[0]:
                    nlr = th[1]
                    break
            if abs(nlr-lr)/lr > 0.0001:
                print ("#"*30)
                print ("swithcing lr",lr,"to", nlr)
                K.set_value( self.discriminator.optimizer.lr, nlr)
                print (K.get_value( self.discriminator.optimizer.lr ))
                K.set_value( self.combined.optimizer.lr, nlr)
                print (K.get_value( self.combined.optimizer.lr ))
                print ("#"*30)

        return np.asarray([epoch_disc_loss, epoch_gen_loss])

    def prepare_geant4_data(self, **args):
        total = 0
        host = os.environ.get('HOST',os.environ.get('HOSTNAME',socket.gethostname()))
        if 'daint' in host:
            sortedpath = '/scratch/snx3000/vlimant/3DGAN/Sorted/sorted_*.hdf5'
        elif 'titan' in host:
            sortedpath = '//ccs/proj/csc291/DATA/3DGAN/sorted/sorted_*.hdf5'
        else:
            sortedpath = '/data/shared/3DGAN/sorted/sorted_*.hdf5'

        m = 2  #number of moments
        var = {}
        energies, srtev = load_sorted(sortedpath)
        for energy in energies:
            var["nevents" + str(energy)]= srtev["ep" + str(energy)].shape[0]
            var["ep"+str(energy)] = srtev["ep" + str(energy)]
            var["ecal_sum"+ str(energy)] = np.sum(srtev["events" + str(energy)], axis = (1, 2, 3))
            sumsx_act, sumsy_act, sumsz_act = get_sums(srtev["events" + str(energy)])
            momentX_act, momentY_act, momentZ_act = get_moments(srtev["events" + str(energy)], sumsx_act, sumsy_act, sumsz_act, var['ecal_sum'+str(energy)], m)
            var['moms_x'+str(energy)]=  np.mean(momentX_act, axis=0)
            var['moms_y'+str(energy)]=  np.mean(momentY_act, axis=0)
            var['moms_z'+str(energy)]=  np.mean(momentZ_act, axis=0)
            var['sumx'+str(energy)], var['sumy'+str(energy)], var['sumz'+str(energy)] = np.mean(sumsx_act, axis=0), np.mean(sumsy_act, axis=0), np.mean(sumsz_act, axis=0)

        return energies, var

    def figure_of_merit(self, **args):
        #print (self.histories)
        delta_loss = np.abs(self.histories['discriminator_model']['classification_loss'][-1] - self.histories['combined_model']['classification_loss'][-1])
        #return delta_loss
        
        if (not self.calculate_fom) :
            raise ValueError('FOM not enabled: No Geant4 data calculated')
        total = 0
        m = 2  #number of moments
        latent= self.latent_size
        energies = self.energies
        g4var = self.g4var 
        ganvar={}
        if not energies:
           raise ValueError('No sorted file found')
        for energy in energies:
            events_gan  = generate(self.generator, g4var["nevents" + str(energy)], latent, g4var["ep" + str(energy)]/100.)
            events_gan[events_gan < 1e-4] = 0.
            ecal_gan  = np.sum(events_gan, axis = (1, 2, 3))
            sumsx_gan, sumsy_gan, sumsz_gan  = get_sums(events_gan)
            momentX_gan, momentY_gan, momentZ_gan = get_moments(events_gan, sumsx_gan, sumsy_gan, sumsz_gan, ecal_gan, m)

            ganvar["moms_x"+ str(energy)]= np.mean(momentX_gan, axis=0)
            ganvar["moms_y"+ str(energy)]= np.mean(momentY_gan, axis=0)
            ganvar["moms_z"+ str(energy)]= np.mean(momentZ_gan, axis=0)
            ganvar['sumx'+str(energy)], ganvar['sumy'+str(energy)], ganvar['sumz'+str(energy)] = np.mean(sumsx_gan, axis=0), np.mean(sumsy_gan, axis=0), np.mean(sumsz_gan, axis=0)

        return metric(ganvar, g4var, energies, m)



class GANModelBuilder(ModelBuilder):
    def __init__(self, c, device_name='cpu',tf=False, weights=None):
        ModelBuilder.__init__(self, c )
        self.tf = tf
        self.weights = weights.split(',') if weights else list([None,None])
        self.device = self.get_device_name(device_name) if self.tf else None
        self.model_parameters={}

    def get_backend_name(self):
        return 'tensorflow'
    
    def set_params(self , **args):
        for k,v in args.items():
            self.model_parameters[k] = v

    def build_model(self, local_session=False):
        m = GANModel(**self.model_parameters)
        if self.weights:
            for mm,w in zip(m.models, self.weights):
                if w: mm.load_weights( w )
        return m

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

class GANBuilder(object):
    def __init__(self, parameters):
        self.parameters = parameters

    def builder(self,*params):
        args = dict(zip([p.name for p in self.parameters],params))
        gmb = GANModelBuilder(None) ## will be set later
        gmb.set_params(**args)
        return gmb

