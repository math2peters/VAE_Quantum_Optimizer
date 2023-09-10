import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Conv1D, Reshape, LeakyReLU, Conv1DTranspose, Flatten, Lambda, Concatenate,Layer, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from keras.callbacks import Callback
from tensorflow.nn import gelu
import tensorflow as tf
from VAE_generator import VAEDataGeneratorKeras
import platform
import sys
import tempfile
import pickle


np.random.seed(69)



def reflection_padding_1d(padding):
    """
    Apply reflection padding to a 1D input tensor.

    Args:
        padding: Integer or tuple of 2 integers, specifying the amount of padding along the width dimension.
                 If a single integer is provided, it specifies the same padding value for both sides.

    Returns:
        A Keras Lambda layer that applies reflection padding to the input tensor.
    """

    def pad_func(x):
        paddings = tf.constant([[0, 0], [padding, padding], [0, 0]])
        return tf.pad(x, paddings, mode='REFLECT')

    return Lambda(pad_func)


class VAE:
    def __init__(self, input_size, latent_dim, beta=0):
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.beta = beta


        self.encoder = self.create_encoder()
        self.decoder = self.create_decoder()
        self.cost_network = self.create_cost_network()

        self.vae_model = self.create_vae()

    def create_encoder(self):
        inputs = Input(shape=(self.input_size, 1))
        x = Conv1D(8, 5, activation=gelu)(inputs)
        #x = layers.BatchNormalization()(x)
        x = Conv1D(16, 5, strides=2, activation=gelu)(x)
        #x = layers.BatchNormalization()(x)
        x = Flatten()(x)
        x = Dense(128, activation=gelu)(x)

        mean = Dense(self.latent_dim)(x)
        log_var = Dense(self.latent_dim)(x)
        return Model(inputs, [mean, log_var], name = 'encoder')

    def create_decoder(self):
        inputs = Input(shape=(self.latent_dim,))
        x = Dense(256, activation=gelu, name='decoder_dense1')(inputs)
        #x = layers.BatchNormalization()(x)
        x = Dense(512, activation=gelu,name='decoder_dense2')(x)
        #x = layers.BatchNormalization()(x)
        x = Dense(16 * 32, activation=gelu, name='decoder_dense3')(x)
        #x = layers.BatchNormalization()(x)
        x = Reshape((32, 16))(x)
        x = Conv1DTranspose(32, 5, strides=2, activation=gelu, padding='same')(x)
        #x = layers.BatchNormalization()(x)
        x = Conv1DTranspose(1, 5, activation='tanh', padding='same')(x)
        
        return Model(inputs, x, name = 'decoder')
    
    def create_cost_network(self):
        inputs = Input(shape=(self.latent_dim,))
        y_loss = Dense(128, activation=gelu, name='cost_dense1')(inputs)
        y_loss = Dropout(.3)(y_loss)
        #y_loss = layers.BatchNormalization()(y_loss)
        y_loss = Dense(64, activation=gelu, name='cost_dense2')(y_loss)
        y_loss = Dropout(.3)(y_loss)
        #y_loss = layers.BatchNormalization()(y_loss)
        y_loss = Dense(32, activation=gelu, name='cost_dense3')(y_loss)
        y_loss = Dropout(.3)(y_loss)
        #y_loss = layers.BatchNormalization()(y_loss)
        y_loss = Dense(1, activation='sigmoid', name='cost_activation')(y_loss)
        
        return  Model(inputs, y_loss, name='cost_network')
    

    def create_vae(self):
        inputs = Input(shape=(self.input_size, 1))
        y_true = Input(shape=(1,))
        mean, log_var = self.encoder(inputs)
        z = Sampling()([mean, log_var])
        y_pred = self.decoder(z)
        y_cost = self.cost_network(z)
        
        y_final = VAELossLayer(name='vae_loss_layer', beta=self.beta)([inputs, y_true, [y_pred, y_cost], mean, log_var, self.cost_network])
        model = Model([inputs, y_true], y_final)
        self.vae_model = model
        return model
    
    def set_trainable_layers(self, layer_to_keep_trainable):
        for layer in self.vae_model.layers:
            #print(layer.name)
            if layer.name in layer_to_keep_trainable:
                print("Training on for {}".format(layer.name))
                layer.trainable = True
            else:
                layer.trainable = False
                
    def set_non_trainable_layers(self, layers_to_make_non_trainable):
        for layer in self.vae_model.layers:
            # If the layer's name is in the list of layers to make non-trainable
            if layer.name in layers_to_make_non_trainable:
                print("Turning off training for {}".format(layer.name))
                layer.trainable = False
            else:
                # Otherwise, keep the layer trainable
                layer.trainable = True



class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(mean))
        return mean + tf.exp(0.5 * log_var) * epsilon

class VAELossLayer(layers.Layer):
    def __init__(self, beta, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)
        self.beta = self.add_weight(name='beta', shape=(), initializer=tf.keras.initializers.Constant(beta), trainable=False)
        self.alpha = self.add_weight(name='alpha', shape=(), initializer=tf.keras.initializers.Constant(1), trainable=False)
        self.gamma = self.add_weight(name='gamma', shape=(), initializer=tf.keras.initializers.Constant(0), trainable=False)
        self.reg = self.add_weight(name='reg', shape=(), initializer=tf.keras.initializers.Constant(1e-3), trainable=False)

    def call(self, inputs):
        reconstruction_true, cost_true, y_pred, z_mean, z_log_var, cost_network = inputs
        
        reconstruction = y_pred[0]
        cost = y_pred[1]
        reconstruction_loss = tf.reduce_mean(tf.square(reconstruction_true - reconstruction), axis=[1, 2]) 
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1) 
        cost_loss = tf.square(cost_true - cost) * self.gamma 
        
        # Calculate the L2 regularization term

        l2_loss = 0
        for layer_name in ['cost_dense1', 'cost_dense2', 'cost_dense3']:
            layer = cost_network.get_layer(name=layer_name)
            weights = layer.kernel  # Get the kernel weights
            l2_loss += self.reg * tf.reduce_sum(tf.square(weights))
        
        loss = reconstruction_loss * self.alpha + kl_loss* self.beta  + cost_loss + l2_loss
        self.add_loss(loss, inputs=inputs)
        self.add_metric(reconstruction_loss, name='reconstruction_loss', aggregation='mean')
        self.add_metric(kl_loss, name='kl_loss', aggregation='mean')
        self.add_metric(cost_loss, name='cost_loss', aggregation='mean')
        self.add_metric(l2_loss, name='l2_loss', aggregation='mean')
        return y_pred  # Return y_pred as output for convenience


class BetaScheduler(Callback):
    def __init__(self, start_beta, end_beta, turn_on_step):
        super(BetaScheduler, self).__init__()
        self.start_beta = start_beta
        self.end_beta = end_beta
        self.turn_on_step = turn_on_step
        self.step = 0

    def on_train_batch_end(self, batch, logs=None):
        if self.step > self.turn_on_step:
            beta_value = self.end_beta 
        else:
            beta_value =self.start_beta
        K.set_value(self.model.get_layer('vae_loss_layer').beta, beta_value)
        self.step += 1
        
class CostScheduler(Callback):
    def __init__(self, total_steps):
        super(CostScheduler, self).__init__()

        self.total_steps = total_steps
        self.step = 0

    def on_train_batch_end(self, batch, logs=None):
        if self.step > self.total_steps/2:
            train_cost = 1
            K.set_value(self.model.get_layer('vae_loss_layer').train_cost, train_cost)
        self.step += 1


   