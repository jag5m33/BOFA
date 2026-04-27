import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from pybofa.prep.config import model_params as mcfg

def run_ssae(x_train, x_test, y_labels):
    input_dim = x_train.shape[1]
    latent_dim = mcfg.latent_dim # Now 6D based on your config

    input_layer = layers.Input(shape=(input_dim,))
    
    # Encoder: Expanded to handle the 6D flow
    x = layers.Dense(16, activation='relu')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(12, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Latent Bottleneck: The 6D Forensic Space
    latent = layers.Dense(
        latent_dim, 
        activation='relu', 
        name='latent',
        activity_regularizer=regularizers.l1(mcfg.l1_reg)
    )(x)
    
    # Decoder: Mirroring the Encoder
    x = layers.Dense(12, activation='relu')(latent)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(16, activation='relu')(x)
    output_layer = layers.Dense(input_dim, activation='linear')(x)
    
    model = models.Model(input_layer, output_layer)
    encoder = models.Model(input_layer, latent)
    
    # STABILIZER: Slow learning rate = Smooth Elbow Plot
    lr_stabilizer = callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.2, patience=10, min_lr=1e-7
    )
    
    # Weighting: "Blinds" the model to suspects
    weights = np.where(y_labels == 1, 0.0001, 1.0)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
        loss='mse'
    )
    
    history = model.fit(
        x_train, x_train, 
        sample_weight=weights, 
        epochs=mcfg.epochs, 
        batch_size=mcfg.batch_size, 
        callbacks=[lr_stabilizer],
        verbose=0, 
        shuffle=True
    )
    
    reconstructed = model.predict(x_test, verbose=0)
    latent_space = encoder.predict(x_test, verbose=0)
    scores = np.mean(np.square(x_test - reconstructed), axis=1)
    
    return scores, latent_space, model, reconstructed, encoder, history