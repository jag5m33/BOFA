import numpy as np
from tensorflow.keras import models, layers, callbacks

def run_ae(train_data, full_data, cfg):
    input_dim = train_data.shape[1]
    
    inputs = layers.Input(shape=(input_dim,))
    # Symmetric architecture
    x = layers.Dense(16, activation='relu')(inputs)
    bottleneck_layer = layers.Dense(cfg.latent_dim, activation='linear', name='bottleneck')(x)
    x = layers.Dense(16, activation='relu')(bottleneck_layer)
    outputs = layers.Dense(input_dim, activation='linear')(x)
    
    autoencoder = models.Model(inputs, outputs)
    # Create an ENCODER model to extract the bottleneck specifically
    encoder = models.Model(inputs, bottleneck_layer)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    
    early_stop = callbacks.EarlyStopping(monitor='loss', patience=cfg.patience, restore_best_weights=True)
    
    autoencoder.fit(train_data, train_data, 
                    epochs=cfg.epochs, 
                    batch_size=cfg.batch_size, 
                    callbacks=[early_stop], verbose=0)
    
    # 1. Get Reconstruction Scores
    recons = autoencoder.predict(full_data)
    scores = np.mean(np.square(full_data - recons), axis=1)
    
    # 2. Extract Latent Space (Bottleneck)
    latent_train = encoder.predict(train_data)
    latent_full = encoder.predict(full_data)
    
    return scores, latent_train, latent_full