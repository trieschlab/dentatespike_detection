import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self,
                 encoder,
                 decoder,
                 kl_ratio,
                 **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.kl_ratio = kl_ratio
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
        
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            mse = keras.losses.MeanSquaredError()
            reconstruction_loss = mse(data, reconstruction)
            kl_loss = self.kl_ratio*-0.5 * (1 + z_log_var - tf.square(z_mean)
                                            - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    

def build_encoder_small(
        enc_units=[512, 512, 256, 128, 2],
        drop_fc=0.5, actfn='relu'):
    """
    Build simple autoencoder with only dense and dropout layers
    
    Params
    ------
    enc_units : list
       Number of units per layer. First/last layer must match input/output.
    drop_fc : float
       Fraction of dropout units
    actfn : str ['relu'], keras activation function

    Returns
    -------
    encoder : keras.Model

    """

    enc_units = np.array(enc_units)
    
    # define input layer
    encoder_inputs = layers.Input(shape=(enc_units[0],))
    x = encoder_inputs
    for n_i in enc_units[1:-2]:
        x = layers.Dense(n_i, activation=actfn)(x)
        x = layers.Dropout(drop_fc)(x)
    x = layers.Flatten()(x)
    
    z_mean = layers.Dense(enc_units[-1], name="z_mean")(x)
    z_log_var = layers.Dense(enc_units[-1], name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(
        encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    return encoder

    
def build_decoder_small(
        dec_units=[2, 128, 256, 512, 512],
        drop_fc=0.5, actfn='relu'):
    """
    Build simple autoencoder with only dense and dropout layers
    
    Params
    ------
    dec_units : list
       Number of units per layer. First/last layer must match input/output.
    drop_fc : float
       Fraction of dropout units
    actfn : str ['relu'], keras activation function

    Returns
    -------
    encoder : keras.Model

    """
    dec_units = np.array(dec_units)
    
    latent_inputs = keras.Input(shape=(dec_units[0],))
    x = latent_inputs
    
    for n_i in dec_units[1:-2]:
        x = layers.Dense(n_i, activation=actfn)(x)
        x = layers.Dropout(drop_fc)(x)
    x = layers.Flatten()(x)
    
    decoder_outputs = layers.Dense(dec_units[-1])(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    return decoder

    
def build_encoder_convolutional(self, latent_dim, X_shape):
    """
    Build the encoder
    """
    encoder_inputs = layers.Input(shape=(X_shape[1], X_shape[2]))
    x = encoder_inputs
    x = layers.Conv1D(
        filters=8, kernel_size=7, padding="same", strides=2, activation="relu"
    )(x)
    x = layers.Conv1D(
        filters=4, kernel_size=7, padding="same", strides=4, activation="relu"
    )(x)
    x = layers.Conv1D(
        filters=2, kernel_size=7, padding="same", strides=4, activation="relu"
    )(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16,)(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(
        encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    return encoder


def build_decoder_convolutional(self, latent_dim, X_shape):
    """
    Build the decoder
    """
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = latent_inputs
    x = layers.Dense(16)(x)
    x = layers.Reshape((8, 2))(x)
    x = layers.Conv1DTranspose(
        filters=4, kernel_size=7, padding="same",
        strides=4, activation="relu"
    )(x)
    x = layers.Conv1DTranspose(
        filters=8, kernel_size=7, padding="same",
        strides=4, activation="relu"
    )(x)
    x = layers.Conv1DTranspose(
        filters=1, kernel_size=7, padding="same",
        strides=2, activation="relu"
    )(x)
    decoder_outputs = layers.Dense(16)(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    return decoder

    
class VAE_old(keras.Model):
    def __init__(
            self, latent_dim, X_shape, **kwargs):
        latent_dim = 2
        super(VAE_old, self).__init__(**kwargs)
        self.encoder = self.build_encoder(latent_dim, X_shape)
        self.decoder = self.build_decoder(latent_dim, X_shape)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def build_encoder(self, latent_dim, X_shape):
        """
        Build the encoder
        """
        encoder_inputs = layers.Input(shape=(X_shape[1], X_shape[2]))
        x = encoder_inputs
        x = layers.Conv1D(
            filters=8, kernel_size=7, padding="same", strides=2
       )(x)
        x = layers.Conv1D(
            filters=4, kernel_size=7, padding="same", strides=4, activation="relu"
       )(x)
        x = layers.Conv1D(
            filters=2, kernel_size=7, padding="same", strides=4, activation="relu"
       )(x)
        x = layers.Flatten()(x)
        x = layers.Dense(16,)(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(
            encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()
        return encoder

    def build_decoder(self, latent_dim, X_shape):
        """
        Build the decoder
        """
        latent_inputs = keras.Input(shape=(latent_dim,))
        x = latent_inputs
        x = layers.Dense(16)(x)
        x = layers.Reshape((8, 2))(x)
        x = layers.Conv1DTranspose(
            filters=4, kernel_size=7, padding="same",
            strides=4, activation="relu"
        )(x)
        x = layers.Conv1DTranspose(
            filters=8, kernel_size=7, padding="same",
            strides=4, activation="relu"
        )(x)
        x = layers.Conv1DTranspose(
            filters=1, kernel_size=7, padding="same",
            strides=2
        )(x)
        decoder_outputs = x
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()
        return decoder

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            mse = keras.losses.MeanSquaredError()
            reconstruction_loss = mse(data, reconstruction)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + self.kl_alpha*kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

