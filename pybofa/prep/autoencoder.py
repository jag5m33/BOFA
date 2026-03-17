#Input → Encoder → Latent space → Decoder → Reconstructed input
# a type of artificial neural network
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.model import Model 
from tensorflow.keras.layers import Input, Dense, Flatten, ReShape
from tensorflow.keras.optimizers import Adam

# you need to perform an 80/20 SPLIT on the dataset to get an x train and x test 

data = # the PREPROCESSED CLEANED AND NORMALISED DATASET OF ENDOCRINE VARIABLES

x_train, x_test = data
#since the model is unsupervised the y train and y test DO NOT EXIST as you arent tring to get an answer for the data itself, you want to map the internal structures in the data
x_train.shape 
x_test.shape 

# the shapes of the training and test sets will tell us what we want to return from the autoencoder - we will use the 2nd and 3rd dimension shape and multiply (from the .shape) as we want the dimensions of one sample to put into the autoencoder one at a time 


encoder.input = keras.Input(x_train)



#the autoencoder model fitting now:

model.fit(x_train, x_test)
