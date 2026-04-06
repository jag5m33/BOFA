import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from pybofa.prep.config import model_params as mcfg
from pybofa.prep.config import ssae as sscfg

def run_ssae(train_data, full_data, labels, epochs):
    """
    labels: 1D array same length as train_data. 
            0 for unknown/general, 1 for GH_CONTROL.
    """
    input_dim = train_data.shape[1]
    init = tf.keras.initializers.GlorotUniform(seed=42)
    hidden_dim = int((input_dim + mcfg.latent_dim) / 2) + 4 

    # --- ARCHITECTURE ---
    inputs = layers.Input(shape=(input_dim,), name="input_layer")
    
    # ENCODER
    x = layers.Dense(hidden_dim, activation='relu', kernel_initializer=init)(inputs)
    x = layers.Dropout(0.1)(x) 
    
    # BOTTLENECK (The shared representation)
    bottleneck = layers.Dense(mcfg.latent_dim, activation='linear', 
                              kernel_initializer=init, name='bottleneck')(x)
    
    # PATH A: DECODER (Reconstruction)
    x_dec = layers.Dense(hidden_dim, activation='relu', kernel_initializer=init)(bottleneck)
    reconstruction_output = layers.Dense(input_dim, activation='linear', 
                                         kernel_initializer=init, name='reconstruction')(x_dec)
    
    # PATH B: CLASSIFIER (Supervision head)
    classifier_output = layers.Dense(1, activation='sigmoid', name='classifier')(bottleneck)

    # --- MODEL DEFINITION ---
    # We define the order here: [0] is reconstruction, [1] is classifier
    ssae = models.Model(inputs=inputs, outputs=[reconstruction_output, classifier_output])
    encoder = models.Model(inputs, bottleneck)
    
    # --- COMPILATION ---
    # Using lists here ensures the order matches the Model definition
    ssae.compile(
        optimizer='adam',
        loss=['mse', 'binary_crossentropy'],
        loss_weights=[sscfg.reconstruction, sscfg.classifier]
    )
    
    # --- PREPARING LABELS & WEIGHTS ---
    # Flatten and ensure float32 for Keras compatibility
    y_labels = np.asarray(labels).astype('float32').reshape((-1, 1))

    # We provide a list of weights matching the output order:
    # 1. Weight 1.0 for everyone on reconstruction
    # 2. Weight 0.0/1.0 for classifier (only known samples nudge the latent space)
    
    sample_weights = [
        np.ones(len(train_data), dtype="float32"), # Weight for reconstruction output
        y_labels.flatten() # Weight for classifier output
    ]

    # --- TRAINING ---
    ssae.fit(
        x=train_data, 
        y=[train_data, y_labels], # Passing as a list matches the output order
        sample_weight=sample_weights, 
        epochs=epochs, 
        batch_size=mcfg.batch_size, 
        verbose=0, 
        shuffle=True
    )
    
    # --- OUTPUTS ---
    # Use verbose=0 here to keep the final output clean
    recons, _ = ssae.predict(full_data, verbose=0)
    scores = np.mean(np.square(full_data - recons), axis=1)
    
    latent_train = encoder.predict(train_data, verbose=0)
    latent_full = encoder.predict(full_data, verbose=0)
    
    return scores, latent_train, latent_full