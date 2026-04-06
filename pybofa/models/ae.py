import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from pybofa.prep.config import model_params as mcfg

# 1. SET GLOBAL SEEDS (Must be at the very top)
np.random.seed(42)
tf.random.set_seed(42)
def run_ae(train_data, full_data, epochs):
    input_dim = train_data.shape[1]
    
    # weights to big - model doesnt learn (explodes) + weights too small: models doesnt learn (stuck at local minima). Glorot/Xavier is a good middle ground for MLPs.
    # This is a mathematically optimized way to pick starting numbers. it picks weights based on the number of inputs and outputs in a layer so that the signal stays strong as it passes through the network.
    init = tf.keras.initializers.GlorotUniform(seed=42)
    
    # Calculate a sensible hidden layer size (between input and latent)
    hidden_dim = int((input_dim + mcfg.latent_dim) / 2) + 4 
        # takes average of input and latent (!5 inputs, 14 hidden, 6 latent, 14 hidden = 15 outputs)

    inputs = layers.Input(shape=(input_dim,))
    
    # ENCODER
    x = layers.Dense(hidden_dim, activation='relu', kernel_initializer=init)(inputs)
    # Adding Dropout can help the SVM/GMM find better boundaries by preventing overfitting
    x = layers.Dropout(0.1)(x) 
    
    bottleneck = layers.Dense(mcfg.latent_dim, activation='linear', 
                              kernel_initializer=init, name='bottleneck')(x)
    
    # DECODER (Mirrors Encoder)
    x = layers.Dense(hidden_dim, activation='relu', kernel_initializer=init)(bottleneck)
    outputs = layers.Dense(input_dim, activation='linear', kernel_initializer=init)(x)
    
    autoencoder = models.Model(inputs, outputs)
    encoder = models.Model(inputs, bottleneck)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    
    autoencoder.fit(
        train_data, train_data, 
        epochs=epochs, 
        batch_size=mcfg.batch_size, 
        verbose=0,
        shuffle=False
    )
    
    recons = autoencoder.predict(full_data)
    scores = np.mean(np.square(full_data - recons), axis=1)
    
    return scores, encoder.predict(train_data), encoder.predict(full_data)