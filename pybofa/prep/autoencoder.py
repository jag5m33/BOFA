#Input → Encoder → Latent space → Decoder → Reconstructed input
# a type of artificial neural network
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.model import Model 
from tensorflow.keras.layers import Input, Dense, Flatten, ReShape
from tensorflow.keras.optimizers import Adam

#Autoencoder works on all data GHadmin. data and the athlete data itself = merged_df 

import os 
import pandas as pd
from config import data as dcfg

df = pd.read_csv(dcfg.gh_admin)


df.shape



#encoder.input = keras.Input(x_train)



#the autoencoder model fitting now:

#model.fit(x_train, x_test)

# Add additional function into the bottom here: define GH-direction - comapred to the total vector of GH and the normal vector - what is the threshold difference in distance between these two points
# use later as secondary evaluater - if distances are closer to the number (called X) then they are far away, if they 
