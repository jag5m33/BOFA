import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from pybofa.prep.config import model_params as mcfg

# 1. SET GLOBAL SEEDS (Must be at the very top)
np.random.seed(42)
tf.random.set_seed(42)

def run_ae(train_data, full_data, epochs):
    """Trains AE and returns scores and latent coordinates with fixed initialization."""
    
    input_dim = train_data.shape[1]
    
    # 2. Use a fixed kernel_initializer for every Dense layer
    # This ensures the 'starting weights' are identical every time you run the script.
    init = tf.keras.initializers.GlorotUniform(seed=42)

    inputs = layers.Input(shape=(input_dim,))
    
    x = layers.Dense(16, activation='relu', kernel_initializer=init)(inputs)
    
    bottleneck = layers.Dense(mcfg.latent_dim, activation='linear', 
                              kernel_initializer=init, name='bottleneck')(x)
    
    x = layers.Dense(16, activation='relu', kernel_initializer=init)(bottleneck)
    
    outputs = layers.Dense(input_dim, activation='linear', kernel_initializer=init)(x)
    
    autoencoder = models.Model(inputs, outputs)
    encoder = models.Model(inputs, bottleneck)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # 3. Disable 'shuffle' or set a seed for the shuffle
    # If the training data is fed in a different order, the model learns slightly differently.
    autoencoder.fit(
        train_data, 
        train_data, 
        epochs=epochs, 
        batch_size=mcfg.batch_size, 
        verbose=0,
        shuffle=False  # Crucial for 100% identical results across runs
    )
    
    recons = autoencoder.predict(full_data)
    
    # Calculate Reconstruction Error (MSE per athlete)
    scores = np.mean(np.square(full_data - recons), axis=1)
    
    return scores, encoder.predict(train_data), encoder.predict(full_data)