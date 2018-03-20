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
from keras.optimizers import RMSprop
#from EcalEnergyGan import generator, discriminator
import numpy as np
import time

import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import (Input, Dense, Reshape, Flatten, Lambda, merge,
                          Dropout, BatchNormalization, Activation, Embedding)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import (UpSampling3D, Conv3D, ZeroPadding3D,
                                        AveragePooling3D)

from mpi_learn.utils import get_device_name
from mpi_learn.train.model import MPIModel, ModelBuilder

def discriminator():

    image = Input(shape=( 25, 25, 25,1 ))

    #x = Conv3D(32, (5, 5,5), data_format='channels_first', padding='same')(image)
    x = Conv3D(32, 5, 5,5,border_mode='same')(image)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)

    x = ZeroPadding3D((2, 2,2))(x)
    #x = Conv3D(8, (5, 5, 5), data_format='channels_first', padding='valid')(x)
    x = Conv3D(8, 5, 5,5,border_mode='same')(image)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = ZeroPadding3D((2, 2, 2))(x)
    #x = Conv3D(8, (5, 5,5), data_format='channels_first', padding='valid')(x)
    x = Conv3D(8, 5, 5,5,border_mode='same')(image)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = ZeroPadding3D((1, 1, 1))(x)
    #x = Conv3D(8, (5, 5, 5), data_format='channels_first', padding='valid')(x)
    x = Conv3D(8, 5, 5,5,border_mode='same')(image)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = AveragePooling3D((2, 2, 2))(x)
    h = Flatten()(x)

    dnn = Model(image, h)

    #image = Input(shape=(1, 25, 25, 25))

    dnn_out = dnn(image)


    fake = Dense(1, activation='sigmoid', name='generation')(dnn_out)
    aux = Dense(1, activation='linear', name='auxiliary')(dnn_out)
    ecal = Lambda(lambda x: K.sum(x, axis=(1, 2, 3)), output_shape=(1,))(image)
    #return Model(input=image, output=[fake, aux, ecal])
    return Model(output=[fake, aux, ecal], input=image)

def generator(latent_size=1024, return_intermediate=False):

    latent = Input(shape=(latent_size, ))

    x = Dense(64 * 7* 7)(latent)
    x = Reshape((7, 7,8, 8))(x)
    x = Conv3D(64, 6, 6, 8, border_mode='same', init='he_uniform')(x)
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
    fake_image = loc(latent)
    Model(input=[latent], output=fake_image)
    return Model(input=[latent], output=fake_image)



def bit_flip(x, prob=0.05):
    """ flips a int array's values with some probability """
    x = np.array(x)
    selection = np.random.uniform(0, 1, x.shape) < prob
    x[selection] = 1 * np.logical_not(x[selection])
    return x


class GANModel(MPIModel):
    def __init__(self, latent_size=10):
        #self.epoch_disc_loss = []
        #self.epoch_gen_loss = []
        print('[INFO] Building generator')
        self.generator = generator(latent_size)
        print('[INFO] Building discriminator')
        self.discriminator = discriminator()
        self.latent_size=latent_size
        self.batch_size= None
        print('[INFO] Building combined')
        latent = Input(shape=(self.latent_size, ), name='combined_z')
        fake_image = self.generator(latent)
        fake, aux, ecal = self.discriminator(fake_image)
        self.combined = Model(
            input=[latent],
            output=[fake, aux, ecal],
            name='combined_model'
        )
        #self.combined.summary()

        MPIModel.__init__(self, models = [ self.generator,
                                           self.discriminator ])

        self.g_cc = 0
        self.d_cc = 0
        self.p_cc = 0
        self.g_t = []
        self.d_t = []
        self.p_t = []
        
    def compile(self, **args):
        ## args are fully ignored here 
        print('[INFO] IN GAN MODEL: COMPILE')       
        #self.generator.summary()
        #self.discriminator.summary()

        self.generator.compile(optimizer=RMSprop(), loss='binary_crossentropy')
        self.discriminator.compile(
             optimizer=RMSprop(),
             loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
             loss_weights=[6, 0.2, 0.1]
        )

        self.discriminator.trainable = False
        self.combined.compile(
           #optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
           optimizer=RMSprop(),
           loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
           loss_weights=[6, 0.2, 0.1]
        ) 


    #def predict(self, generator_input):
    #    return self.generator.predict(generator_input)
        
    def batch_transform(self, x, y):
        x_disc_real =x
        y_disc_real = y
        if self.batch_size is None:
            ## fix me, maybe
            self.batch_size = x_disc_real.shape[0]
            print ("initializing sizes",x_disc_real.shape,[ yy.shape for yy in y])

        noise = np.random.normal(0, 1, (self.batch_size, self.latent_size))
        sampled_energies = np.random.uniform(1, 5,(self.batch_size,1))
        generator_ip = np.multiply(sampled_energies, noise)
        ecal_ip = np.multiply(2, sampled_energies)
        now = time.mktime(time.gmtime())
        if self.p_cc>1 and len(self.p_t)%100==0:
            print ("prediction average",np.mean(self.p_t),"[s]' over",len(self.p_t))
        generated_images = self.generator.predict(generator_ip)
        done = time.mktime(time.gmtime())
        if self.p_cc:
            self.p_t.append( done - now )
        self.p_cc +=1

        ## need to bit flip the true labels too
        bf = bit_flip(y[0])
        y_disc_fake = [bit_flip(np.zeros(self.batch_size)), sampled_energies.reshape((-1,)), ecal_ip.reshape((-1,))]
        re_y = [bf, y_disc_real[1], y_disc_real[2]]

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


        noise = np.random.normal(0, 1, (2*self.batch_size, self.latent_size))
        sampled_energies = np.random.uniform(1, 5, (2*self.batch_size,1 ))
        generator_ip = np.multiply(sampled_energies, noise)
        ecal_ip = np.multiply(2, sampled_energies)
        trick = np.ones(2*self.batch_size)        

        X_for_combined = [generator_ip]
        Y_for_combined = [trick,sampled_energies.reshape((-1, 1)), ecal_ip]
        
        return (X_for_disc,Y_for_disc,X_for_combined,Y_for_combined)

    def test_on_batch(self,x, y, sample_weight=None):
        (X_for_disc,Y_for_disc,X_for_combined,Y_for_combined) = self.batch_transform(x,y)
        epoch_disc_loss = self.discriminator.test_on_batch(X_for_disc,Y_for_disc)
        epoch_gen_loss = self.combined.test_on_batch(X_for_combined,Y_for_combined)

        return np.asarray([epoch_disc_loss, epoch_gen_loss])

    def train_on_batch(self, x, y,
                   sample_weight=None,
                   class_weight=None):

        (X_for_disc,Y_for_disc,X_for_combined,Y_for_combined) = self.batch_transform(x,y)
        if self.d_cc>1 and len(self.d_t)%100==0:
            print ("discriminator average",np.mean(self.d_t),"[s] over ",len(self.d_t))
        now = time.mktime(time.gmtime())
        epoch_disc_loss = self.discriminator.train_on_batch(X_for_disc,Y_for_disc)
        done = time.mktime(time.gmtime())
        if self.d_cc:
            self.d_t.append( done - now )
        self.d_cc+=1
        if self.g_cc>1 and len(self.g_t)%100==0:
            print ("generator average ",np.mean(self.g_t),"[s] over",len(self.g_t))
        now = time.mktime(time.gmtime())
        epoch_gen_loss = self.combined.train_on_batch(X_for_combined,Y_for_combined)
        done = time.mktime(time.gmtime())
        if self.g_cc:
            self.g_t.append( done - now )
            
        return np.asarray([epoch_disc_loss, epoch_gen_loss])
    
    def old_train_on_batch(self, x, y,
                   sample_weight=None,
                   class_weight=None):

        x_disc_real =x
        y_disc_real = y
        if self.batch_size is None:
            ## fix me, maybe 
            self.batch_size = x_disc_real.shape[0]
            print ("initializing sizes",x_disc_real.shape,[ yy.shape for yy in y])
            
            
        noise = np.random.normal(0, 1, (self.batch_size, self.latent_size))
        sampled_energies = np.random.uniform(1, 5,(self.batch_size,1))
        generator_ip = np.multiply(sampled_energies, noise)
        ecal_ip = np.multiply(2, sampled_energies)
        now = time.mktime(time.gmtime())
        if self.p_cc>1 and len(self.p_t)%100==0:
            print ("prediction average",np.mean(self.p_t),"[s]' over",len(self.p_t))
        generated_images = self.generator.predict(generator_ip)
        done = time.mktime(time.gmtime())
        if self.p_cc:
            self.p_t.append( done - now )
        self.p_cc +=1
        
        ## need to bit flip the true labels too
        bf = bit_flip(y[0])
        y_disc_fake = [bit_flip(np.zeros(self.batch_size)), sampled_energies.reshape((-1,)), ecal_ip.reshape((-1,))]
        re_y = [bf, y_disc_real[1], y_disc_real[2]]
        #print ("got",[yy.shape for yy in re_y])
        #print ("got",[yy.shape for yy in y_disc_fake])
        
        two_pass = False
        #print ("calling discr",self.d_cc)
        if self.d_cc>1 and len(self.d_t)%100==0:
            print ("discriminator average",np.mean(self.d_t),"[s] over ",len(self.d_t))
        now = time.mktime(time.gmtime())
        if two_pass:

            #real_batch_loss = self.discriminator.train_on_batch(x_disc_real, y_disc_real)
            real_batch_loss = self.discriminator.train_on_batch(x_disc_real,re_y)
            fake_batch_loss = self.discriminator.train_on_batch(generated_images, y_disc_fake)
            epoch_disc_loss = [
                (a + b) / 2 for a, b in zip(real_batch_loss, fake_batch_loss)
            ]
        else:
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
            epoch_disc_loss = self.discriminator.train_on_batch( bb_x, bb_y )


        done = time.mktime(time.gmtime())
        if self.d_cc:
            self.d_t.append( done - now )
        self.d_cc+=1


        #print ("calling gen",self.g_cc)
        if self.g_cc>1 and len(self.g_t)%100==0:
            print ("generator average ",np.mean(self.g_t),"[s] over",len(self.g_t))
        now = time.mktime(time.gmtime())
        if two_pass:
            gen_losses = []
            trick = np.ones(self.batch_size)        
            for _ in range(2):
                noise = np.random.normal(0, 1, (self.batch_size, self.latent_size))
                sampled_energies = np.random.uniform(1, 5, ( self.batch_size,1 ))
                generator_ip = np.multiply(sampled_energies, noise)
                ecal_ip = np.multiply(2, sampled_energies)
                
                gen_losses.append(self.combined.train_on_batch(
                    [generator_ip],
                    [trick, sampled_energies.reshape((-1, 1)), ecal_ip]))

            epoch_gen_loss = [
                (a + b) / 2 for a, b in zip(*gen_losses)
            ]
        else:
            noise = np.random.normal(0, 1, (2*self.batch_size, self.latent_size))
            sampled_energies = np.random.uniform(1, 5, (2*self.batch_size,1 ))
            generator_ip = np.multiply(sampled_energies, noise)
            ecal_ip = np.multiply(2, sampled_energies)
            trick = np.ones(2*self.batch_size)
            epoch_gen_loss = self.combined.train_on_batch(
                [generator_ip],
                [trick,sampled_energies.reshape((-1, 1)), ecal_ip])

            
        
        done = time.mktime(time.gmtime())
        if self.g_cc:
            self.g_t.append( done - now )
        self.g_cc+=1
        
        #self.epoch_disc_loss.extend( epoch_disc_loss )
        #self.epoch_gen_loss.extend( epoch_gen_loss )
        return np.asarray([epoch_disc_loss, epoch_gen_loss])



class GANModelBuilder(ModelBuilder):
    def __init__(self, c, device_name='cpu',tf=False):
        ModelBuilder.__init__(self, c)
        self.tf = tf
        self.device = self.get_device_name(device_name) if self.tf else None
        
    def build_model(self):
        if self.tf:
            m = GANModel()
            return m
        else:
            m = GANModel()
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
