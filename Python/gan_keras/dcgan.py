# Author: Hamaad Musharaf Shah.

import math
import inspect

import tensorflow

from six.moves import range

import os
import math
import sys
import importlib

import numpy as np

import pandas as pd

from sklearn import linear_model
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.metrics import roc_auc_score

from scipy.stats import norm

import keras
from keras import backend as bkend
from keras import layers
from keras.layers import Input, Dense, BatchNormalization, Dropout, Flatten, convolutional, pooling, Reshape, Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras import metrics
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.utils.generic_utils import Progbar
from keras.preprocessing import image

import tensorflow as tf
from tensorflow.python.client import device_lib

from gan_keras.loss_history import LossHistory

class DeepConvGenAdvNet(BaseEstimator, 
                        TransformerMixin):
    def __init__(self, 
                 z_size=None,
                 iterations=None,
                 batch_size=None):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        
        for arg, val in values.items():
            setattr(self, arg, val)
        
        # Build the discriminator.
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(optimizer=RMSprop(lr=0.0002, 
                                                     clipvalue=1.0,
                                                     decay=1e-8),
                                   loss="binary_crossentropy")

        # Build the generator to fool the discriminator.
        # Freeze the discriminator here.
        self.discriminator.trainable = False
        self.generator = self.build_generator()
        GAN_input = Input(shape=(self.z_size,))
        GAN_fake = self.generator(GAN_input)
        GAN_output = self.discriminator(GAN_fake)

        # Build the GAN.
        self.GAN = Model(GAN_input, GAN_output)
        self.GAN.compile(optimizer=RMSprop(lr=0.0004, 
                                           clipvalue=1.0,
                                           decay=1e-8),
                         loss="binary_crossentropy")
 
    def fit(self,
            X,
            y=None):
        num_train = X.shape[0]
        start = 0
        
        for step in range(self.iterations):
            # Generate a new batch of noise...
            noise = np.random.uniform(low=-1.0, high=1.0, size=(self.batch_size, self.z_size))
            # ...and generate a batch of fake images.
            generated_images = self.generator.predict(noise)
            
            stop = start + self.batch_size
            # Get a batch of real images.
            image_batch = X[start:stop]

            # [real, fake].
            x = np.concatenate((image_batch, generated_images))
            # [real, fake].
            y = np.concatenate([np.ones(shape=(self.batch_size, 1)), np.zeros(shape=(self.batch_size, 1))])
            y += 0.05 * np.random.random(size=y.shape)

            # See if the discriminator can figure itself out.
            self.d_loss = self.discriminator.train_on_batch(x, y)

            # Make new noise.
            noise = np.random.uniform(low=-1.0, high=1.0, size=(self.batch_size, self.z_size))

            # We want to train the generator to trick the discriminator.
            # For the generator, we want all the [real, fake] labels to say real.
            trick = np.ones(shape=(self.batch_size, 1))

            self.gan_loss = self.GAN.train_on_batch(noise, trick)
                
            start += self.batch_size
            if start > num_train - self.batch_size:
                start = 0
            
            if step % 100 == 0:
                print("Step:", step)
                print("Discriminator loss:", self.d_loss)
                print("GAN loss:", self.gan_loss)
                
                img = image.array_to_img(generated_images[0] * 255.0, scale=False)
                img.save("outputs/generated_image" + str(step) + ".png")
                
                img = image.array_to_img(image_batch[0] * 255.0, scale=False)
                img.save("outputs/real_image" + str(step) + ".png")

        return self

    def transform(self,
                  X):
        return self.feature_extractor.predict(X)

    def build_generator(self):
        # We will map z, a latent vector, to image space (..., 28, 28, 1).
        latent = Input(shape=(self.z_size,))

        # This produces a (..., 7, 7, 128) shaped tensor.
        cnn = Dense(units=1024, activation="tanh")(latent)
        cnn = Dense(units=128 * 7 * 7, activation="tanh")(cnn)
        cnn = BatchNormalization()(cnn)
        cnn = Reshape((7, 7, 128))(cnn)

        # Upsample to (..., 14, 14, 64).
        cnn = layers.Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), padding="same", activation="tanh")(cnn)
        cnn = layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding="same", activation="tanh")(cnn)

        # Upsample to (..., 28, 28, 64).
        cnn = layers.Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), padding="same", activation="tanh")(cnn)

        # Take a channel axis reduction to (..., 28, 28, 1).
        fake_img = Conv2D(filters=1, kernel_size=(5, 5), strides=(1, 1), padding="same", activation="sigmoid", kernel_initializer="glorot_normal", name="generator")(cnn)

        return Model(latent, fake_img)
    
    def build_discriminator(self):
        image = Input(shape=(28, 28, 1))

        cnn = Conv2D(filters=64, kernel_size=(5, 5), padding="same", strides=(2, 2), activation="tanh")(image)
        cnn = layers.MaxPooling2D(pool_size=(2, 2))(cnn)
        cnn = Conv2D(filters=128, kernel_size=(5, 5), padding="same", strides=(2, 2), activation="tanh")(cnn)
        cnn = layers.MaxPooling2D(pool_size=(2, 2))(cnn)
        cnn = Flatten()(cnn)
        cnn = Dense(units=1024, activation="tanh")(cnn)
        self.feature_extractor = Model(image, cnn)

        is_real_img = Dense(units=1, activation="sigmoid", name="discriminator")(cnn)     
        
        return Model(image, is_real_img)
            
    def evaluate(self,
                 X):
        num_test = X.shape[0]

        # Generate a new batch of noise.
        noise = np.random.uniform(low=-1.0, high=1.0, size=(num_test, self.z_size))
        generated_images = self.generator.predict(noise)

        # [real, fake].
        x = np.concatenate((X, generated_images))
        # [real, fake].
        y = np.concatenate([np.ones(shape=(num_test, 1)), np.zeros(shape=(num_test, 1))])
        y += 0.05 * np.random.random(size=y.shape)

        self.d_test_loss = self.discriminator.evaluate(x, y)

        # Make new noise.
        noise = np.random.uniform(low=-1.0, high=1.0, size=(num_test, self.z_size))
        trick = np.ones(shape=(num_test, 1))

        self.gan_test_loss = self.GAN.evaluate(noise, trick)
        
        return [self.d_test_loss, self.gan_test_loss]
    
class DeeperConvGenAdvNet(DeepConvGenAdvNet):
    def __init__(self,
                 z_size=None,
                 iterations=None,
                 batch_size=None):
        super(DeeperConvGenAdvNet, self).__init__(z_size=z_size,
                                                  iterations=iterations,
                                                  batch_size=batch_size)
        
    def build_discriminator(self):
        image = Input(shape=(28, 28, 1))

        cnn = Conv2D(filters=100, kernel_size=(8, 8), padding="same", strides=(1, 1), activation="elu")(image)
        cnn = Dropout(rate=0.5)(cnn)
        cnn = Conv2D(filters=100, kernel_size=(8, 8), padding="same", strides=(1, 1), activation="elu")(cnn)
        cnn = Dropout(rate=0.5)(cnn)
        cnn = Conv2D(filters=100, kernel_size=(8, 8), padding="same", strides=(1, 1), activation="elu")(cnn)
        cnn = layers.MaxPooling2D(pool_size=(4, 4))(cnn)
        cnn = Flatten()(cnn)
        self.feature_extractor = Model(image, cnn)

        is_real_img = Dense(units=1, activation="sigmoid", name="discriminator")(cnn)     
        
        return Model(image, is_real_img)
    
class DeepConvGenAdvNetInsurance(DeepConvGenAdvNet):
    def __init__(self,
                 z_size=None,
                 iterations=None,
                 batch_size=None):
        super(DeepConvGenAdvNetInsurance, self).__init__(z_size=z_size,
                                                         iterations=iterations,
                                                         batch_size=batch_size)
        
    def build_generator(self):
        # We will map z, a latent vector, to image space (..., 4, 3, 1).
        latent = Input(shape=(self.z_size,))

        # This produces a (..., 4, 3, 1) shaped tensor.
        cnn = Dense(units=100, activation="tanh")(latent)
        cnn = Dense(units=100, activation="tanh")(cnn)
        cnn = Dense(units=100, activation="tanh")(cnn)
        cnn = Dense(units=1 * 3 * 4, activation="sigmoid")(cnn)
        fake_input_ = Reshape((4, 3, 1))(cnn)

        return Model(latent, fake_input_)
        
    def build_discriminator(self):
        input_ = Input(shape=(4, 3, 1))

        cnn = Flatten()(input_)
        cnn = Dense(units=100, activation="elu")(cnn)
        cnn = Dropout(rate=0.5)(cnn)
        self.feature_extractor = Model(input_, cnn)

        is_real_input_ = Dense(units=1, activation="sigmoid", name="discriminator")(cnn)     
        
        return Model(input_, is_real_input_)