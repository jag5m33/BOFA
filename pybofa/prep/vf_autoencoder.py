#Autoencoder works on all data GHadmin. data and the athlete data itself = merged_df 

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd

import pandas as pd
from pybofa.prep.config import data as dcfg
from pybofa.prep.config import model as mcfg
from pybofa.prep.config import processor as pcfg
import matplotlib.pyplot as plt
import numpy as np

#import matplotlib.pyplot as plt
def proc(merged_df_path): 
    
    df = pd.read_csv(merged_df_path)
    
    ids = df['id']
    sources = df['source']
    
    numeric_df = df.drop(columns=["id", "source"])
    
    # Split BEFORE scaling
    athlete_mask = df['source'] == 'ATHLETE_REF'
    athlete_df = numeric_df[athlete_mask]
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    # Fit ONLY on athletes
    scaler.fit(athlete_df.values)
    
    # Transform ALL data using athlete scaling
    df_tensor = scaler.transform(numeric_df.values).astype('float32')
    
    n_features = df_tensor.shape[1]
    
    return df_tensor, n_features, ids, sources
## BUILDING
def autoenc_build(input_shape, dim):
    model = models.Sequential([
        # Encoder part
        layers.InputLayer(input_shape=(input_shape,), name='input_layer'),

        layers.Dense(32, activation="relu"),
        #layers.Dropout(0.1),
        layers.Dense(16, activation="relu"),
        layers.Dense(dim,
                     name='bottleneck',
                     activity_regularizer=tf.keras.regularizers.l1(1e-3)
                     ), # creates sparrse latent space and better seperation

        # Decoder
        layers.Dense(16, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(input_shape, activation="linear") # Back to 4 features
    ])
    return model

## COMPILING 
def compiler(autoencoder):
    autoencoder.compile(
        optimizer="Adam",
        loss='mse',
        metrics=['mae'] 
    )

## TRAINING
def train_autoencoder(autoencoder, data, epochs=None):
    if epochs is None:
        epochs = mcfg.epochs

    noise_factor = 0.3 # increases feature learning + less memorisation
    noisy_data = data + noise_factor * np.random.normal(size=data.shape)

    early_stop = EarlyStopping(
        monitor = 'loss',
        patience = mcfg.patience,
        restore_best_weights = True
    )

    return autoencoder.fit(
        x=noisy_data,
        y=data,
        epochs=epochs,
        callbacks = [early_stop],
        #shuffle=True,
        verbose=1
    )

def elbow_plot( input_shape, dim, data):
    
    dims = range(1, dim+1)
    errors = []
    for i in dims:
        model = autoenc_build(input_shape, dim=i)
        compiler(model)

        history = train_autoencoder(model, data.copy(), epochs=mcfg.epochs)
        final_loss_value = history.history['loss'][-1]
        errors.append(final_loss_value)
    return(dims, errors)

def plotting(dims, errors):
    plt.figure(figsize=(8, 5))
    plt.plot(dims, errors, color='b', linestyle='-', marker='o') 
    plt.title("Autoencoder Elbow Plot")
    plt.xlabel("Number of Dimensions")
    plt.ylabel("Reconstruction Error (MSE)")
    plt.grid(True)
    plt.show()
    
# to generate 3D co-oridnates all the other 3 machine learning models must use
    # feature extraction strategy: to use latent space - treat autoencoder and encoder 
    # as seperate 'entities'
        #the encoder (a sub model) stops at bottleneck layer - pass data through submodel to generate 3D co-ordinates (latent featues)
        #in keras you can create a new model by defining its input as the origonal models inout and its output 

def latent_space(autoencoder):
    # This reaches directly into the first layer to find the definition
    encoder_input = autoencoder.layers[0].input
    
    # This stays the same
    bottleneck_output = autoencoder.get_layer('bottleneck').output
    
    encoder_model = models.Model(inputs=encoder_input, outputs=bottleneck_output)
    print(f"Encoder created. Latent dimensions: {bottleneck_output.shape[1]}")
    
    return encoder_model, bottleneck_output.shape[1]

def arrays(encoder, data):
    # Generates the 3D coordinates (z1, z2, z3)
    latent_data = encoder.predict(data)
    return latent_data

def splits_for_pipeline(latent_df):
    # --- UPDATE: Use 'source' label instead of ID string matching ---
    # This is much cleaner and matches your R-script labels
    gh_mask = (latent_df['source'] == 'GH_CONTROL')
    
    athlete_data = latent_df[~gh_mask].copy()
    gh_val_data = latent_df[gh_mask].copy()
    
    print(f"Athlete training set: {len(athlete_data)} samples")
    print(f"GH validation set: {len(gh_val_data)} samples")
    
    return athlete_data, gh_val_data
