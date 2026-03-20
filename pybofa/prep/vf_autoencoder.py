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
def proc(merged_df): # Use the path from your config file
    df = pd.read_csv(dcfg.merged_df)
    
    # 1. Isolate IDs and encoded Methods (Column 0, 5) 
    ids = df.iloc[:, 0]
    methods = df.iloc[:, 5]
    
    # 2. Drop the "id" column to leave  4 metrics
    numeric_df = df.drop(columns=["id"])
    
    # 3. Convert to Tensor then to NumPy float32
    df_tensor = tf.convert_to_tensor(numeric_df, dtype=tf.float32).numpy()
    
    # 4. Get the number of features = (4)
    n_features = df_tensor.shape[1]
    
    print(f"Data ready. Rows: {df_tensor.shape[0]}, Features: {n_features}")
    
    return df_tensor, n_features, ids, methods
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
    input = autoencoder.layers[0].input
    vectors = autoencoder.get_layer('bottleneck').output
    dimensions = vectors[:, 0:3]
    
    encoder = models.Model(inputs=input, outputs=dimensions)
    print(f" shape of bottleneck latent space layer of the autoencoder:{dimensions} ")

    
    return encoder, dimensions  # by specialiseing encoder - you create a tool that can run to say encoder.predict(merged_Df) and this will create a numpy array ready for the models downstream


#creating latent_data - a numpy 3D array 
def arrays(encoder, data ):
    latent_data = encoder.predict(data)
    return latent_data
