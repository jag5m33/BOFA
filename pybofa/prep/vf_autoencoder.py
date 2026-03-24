#Autoencoder works on all data GHadmin. data and the athlete data itself = merged_df 

import tensorflow as tf
from tensorflow.keras import models, layers
import pandas as pd

import pandas as pd
from pybofa.prep.config import data as dcfg
from pybofa.prep.config import model as mcfg
from pybofa.prep.config import processor as pcfg
import matplotlib.pyplot as plt

#import matplotlib.pyplot as plt
def proc(merged_df_path): 
    df = pd.read_csv(merged_df_path)
    
    # 1. Isolate Metadata (ID and Source)
    # We keep 'source' separate so the model doesn't use it for training
    ids = df['id']
    sources = df['source']
    
    # 2. Prepare Numeric Data
    # We drop 'id' and 'source' to leave only: sex, age, avg_pnp, avg_igf
    numeric_df = df.drop(columns=["id", "source"])
    
    # 3. Convert to NumPy float32 for TensorFlow
    df_tensor = numeric_df.values.astype('float32')
    
    # 4. Get the number of features (should be 4)
    n_features = df_tensor.shape[1]
    
    print(f"Data ready. Rows: {df_tensor.shape[0]}, Features: {n_features}")
    
    # Return sources as well so they can be mapped back later
    return df_tensor, n_features, ids, sources

## BUILDING
def autoenc_build(input_shape, dim):
    model = models.Sequential([
        # Encoder part
        layers.InputLayer(input_shape=(input_shape,), name = 'input_layer'), 
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(dim, name='bottleneck'), # Your 3D coordinates
        
        # Decoder part (Note: these must be inside the [ ] brackets)
        layers.Dense(32, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(128, activation="relu"),
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
def train_autoencoder(autoencoder, data,  epochs = None):
    # If you didn't pass a specific number, use the config file
    if epochs is None:
        epochs = mcfg.epochs 

    return autoencoder.fit(
        x=data, 
        y=data, 
        epochs=epochs, # Use the 'epochs' variable we just set
        validation_split=pcfg.validation_split
    )

def elbow_plot( input_shape, dim, data):
    dims = range(1, dim+1)
    errors = []
    for i in dims:
        model = autoenc_build(input_shape, dim=i)
        compiler(model)

        history = train_autoencoder(model, data, epochs = mcfg.epochs)
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
