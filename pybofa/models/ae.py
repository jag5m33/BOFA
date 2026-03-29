from tensorflow.keras import models, layers, callbacks
import numpy as np

def build_and_train_ae(train_data, latent_dim, epochs, batch_size, patience):
    input_dim = train_data.shape[1]
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(16, activation='relu')(inputs)
    bottleneck = layers.Dense(latent_dim, name='bottleneck')(x) 
    x = layers.Dense(16, activation='relu')(bottleneck)
    outputs = layers.Dense(input_dim, activation='linear')(x)
    
    autoencoder = models.Model(inputs, outputs)
    encoder = models.Model(inputs, bottleneck)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    early_stop = callbacks.EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)
    autoencoder.fit(train_data, train_data, epochs=epochs, batch_size=batch_size, callbacks=[early_stop], verbose=0)
    
    return autoencoder, encoder

def get_recon_error(model, data):
    recons = model.predict(data)
    return np.mean(np.square(data - recons), axis=1)