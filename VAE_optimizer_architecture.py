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

@tf.function
def predictor_activation(x):
    return tf.where(x < 0, 
                    tf.math.exp(x), 
                    x + 1)

class VAE:
    """
    Variational auto-encoder architecture with an additional network interfaced with the latent space that predicts the outcome the experiment given the latent space as input
    """
    def __init__(self, input_size, latent_dim, beta=0):
        """init

        Args:
            input_size (int): input size of vae 
            latent_dim (int): latent dimension size
            beta (float, optional): beta value for vae. Defaults to 0.
        """
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.beta = beta


        self.encoder = self.create_encoder()
        self.decoder = self.create_decoder()
        self.population_predictor_network = self.create_population_predictor_network()

        self.vae_model = self.create_vae()

    def create_encoder(self):
        # batch norm disabled because of worsened performance
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
    
    def create_population_predictor_network(self, dropout=0.01):
        inputs = Input(shape=(self.latent_dim,))
        y_loss = Dense(256, activation=gelu, name='population_predictor_dense1')(inputs)
        y_loss = Dropout(dropout)(y_loss)
        #y_loss = layers.BatchNormalization()(y_loss)
        y_loss = Dense(256, activation=gelu, name='population_predictor_dense2')(y_loss)
        y_loss = Dropout(dropout)(y_loss)
        #y_loss = layers.BatchNormalization()(y_loss)
        y_loss = Dense(256, activation=gelu, name='population_predictor_dense3')(y_loss)
        y_loss = Dropout(dropout)(y_loss)
        y_loss = Dense(256, activation=gelu, name='population_predictor_dense4')(y_loss)
        y_loss = Dropout(dropout)(y_loss)
        #y_loss = layers.BatchNormalization()(y_loss)
        y_loss = Dense(1, activation=predictor_activation, name='population_predictor_activation')(y_loss)
        
        return  Model(inputs, y_loss, name='population_predictor_network')
    

    def create_vae(self):
        inputs = Input(shape=(self.input_size, 1))
        y_true = Input(shape=(1,))
        mean, log_var = self.encoder(inputs)
        z = Sampling()([mean, log_var])
        y_pred = self.decoder(z)
        y_population_predictor = self.population_predictor_network(z)
        
        y_final = VAELossLayer(name='vae_loss_layer', beta=self.beta)([inputs, y_true, [y_pred, y_population_predictor], mean, log_var, self.population_predictor_network])
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
    """
    Sample latent space with a gaussian distribution"""
    def call(self, inputs):
        mean, log_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(mean))
        return mean + tf.exp(0.5 * log_var) * epsilon

class VAELossLayer(layers.Layer):
    """Custom loss layer for VAE/predictor. Has regularization for the predictor network, and a beta scheduler for the VAE loss

    Args:
        layers (_type_): _description_
    """
    def __init__(self, beta, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)
        self.beta = self.add_weight(name='beta', shape=(), initializer=tf.keras.initializers.Constant(beta), trainable=False)
        self.alpha = self.add_weight(name='alpha', shape=(), initializer=tf.keras.initializers.Constant(1), trainable=False)
        self.gamma = self.add_weight(name='gamma', shape=(), initializer=tf.keras.initializers.Constant(0), trainable=False)
        self.reg = self.add_weight(name='reg', shape=(), initializer=tf.keras.initializers.Constant(1e-5), trainable=False)

    def call(self, inputs):
        reconstruction_true, population_predictor_true, y_pred, z_mean, z_log_var, population_predictor_network = inputs
        
        reconstruction = y_pred[0]
        population_predictor = y_pred[1]
        reconstruction_loss = tf.reduce_mean(tf.square(reconstruction_true - reconstruction), axis=[1, 2]) 
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1) 
        population_predictor_loss = tf.square(population_predictor_true - population_predictor) * self.gamma 
        
        # Calculate the L2 regularization term

        l2_loss = 0
        for layer_name in ['population_predictor_dense1', 'population_predictor_dense2', 'population_predictor_dense3', 'population_predictor_dense4']:
            layer = population_predictor_network.get_layer(name=layer_name)
            weights = layer.kernel  # Get the kernel weights
            l2_loss += self.reg * tf.reduce_sum(tf.square(weights))
        
        loss = reconstruction_loss * self.alpha + kl_loss* self.beta  + population_predictor_loss + l2_loss
        self.add_loss(loss, inputs=inputs)
        self.add_metric(reconstruction_loss, name='reconstruction_loss', aggregation='mean')
        self.add_metric(kl_loss, name='kl_loss', aggregation='mean')
        self.add_metric(population_predictor_loss, name='population_predictor_loss', aggregation='mean')
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
        
class PredictorScheduler(Callback):
    def __init__(self, total_steps):
        super(PredictorScheduler, self).__init__()

        self.total_steps = total_steps
        self.step = 0

    def on_train_batch_end(self, batch, logs=None):
        if self.step > self.total_steps/2:
            train_population_predictor = 1
            K.set_value(self.model.get_layer('vae_loss_layer').train_population_predictor, train_population_predictor)
        self.step += 1


   