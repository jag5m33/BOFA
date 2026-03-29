from tensorflow.keras import models, layers, callbacks
import numpy as np

#dimensionality reduction + reconstruction error for anomaly detection

def build_and_train_ae(train_data, latent_dim, epochs, batch_size, patience):
    # encoder: compress 5D input (age, sex, igf, pnp, ratio) into 3D latent space (removes noise to find biological sign.)

    input_dim = train_data.shape[1]
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(16, activation='relu')(inputs)
    bottleneck = layers.Dense(latent_dim, name='bottleneck')(x) 

    #decoder: tries to reconstruct the origonal data from 3D representation
    x = layers.Dense(16, activation='relu')(bottleneck)
    outputs = layers.Dense(input_dim, activation='linear')(x)
    
    autoencoder = models.Model(inputs, outputs)
        # models.Model: command that assembles NN. (defined ae with laters input, hidden, bottleneck, output) 
        # take those layers and conect them into an object to be employed later for learning 
        
    encoder = models.Model(inputs, bottleneck)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    early_stop = callbacks.EarlyStopping(monitor='loss', 
                                         patience=patience, 
                                         restore_best_weights=True)
    autoencoder.fit(train_data, 
                    train_data, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    allbacks=[early_stop], # prevent overfitting by stopping trainig when model stops getting better at rep. normal athletes)
                    verbose=0)
    
    return autoencoder, encoder

#calcualtes how much the mdoel 'struggled' to rebuilt sample
def get_recon_error(model, data):
        # if athlete is doped = their biological is unfamiliar to model (reconstruction error here should be high)
    recons = model.predict(data)

    return np.mean(np.square(data - recons), axis=1)
    #data - recons = residual (find sthe raw difference between athlete true biomarkers and what model expects to see)
    # np.square = square the residuals to penalise larger errors more heavily (doped athletes should have much higher error)
    # np.mean(..., axis=1) = average the error across all features to get a single recon error score per athlete (higher score = more likely to be doped)
