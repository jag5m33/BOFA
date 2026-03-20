

from pybofa.prep.config import model as mcfg
from pybofa.prep.config import data as dcfg
from pybofa.prep.config import isolation_forest as icfg
 
import pybofa.prep.vf_autoencoder as aut
import pybofa.models.IF as IF 
import numpy as np
import pandas as pd
import tensorflow as tf


# by saying df_tensor and n_features the definition pulls out the returns and put them in their correspnding variable names 
df_tensor, n_features, ids, methods = aut.proc(dcfg.merged_df) 

#run the plotting first to prove that 3 is the best number of dimesions
dims, final_loss_values = aut.elbow_plot(n_features, dim = mcfg.dim, data = df_tensor)
aut.plotting(dims, final_loss_values)

#build autoenc.
autoencoder = aut.autoenc_build(n_features, 3)
aut.compiler(autoencoder)
history = aut.train_autoencoder(autoencoder, data = df_tensor, epochs = mcfg.epochs)

autoencoder.save('trained_bofa_autoencoder.keras') 
print("Model frozen. Coordinates will be stable from now on.") 


# #setting up the latent space & saving it
autoencoder = tf.keras.models.load_model('trained_bofa_autoencoder.keras')

encoder, dimensions = aut.latent_space(autoencoder)

print(f"Total Athletes: {latent_data.shape[0]}")
print(f"Coordinates per Athlete: {latent_data.shape[1]}")
print(f"Full Shape Tuple: {latent_data.shape}")

# #plot  t-SNE 
latent_data = IF.t_sne(latent_data)

 #saving latent data 
latent_full_df = pd.DataFrame(latent_data, columns=['z1', 'z2', 'z3'])
latent_full_df['id'] = ids.values
latent_full_df.to_csv('athlete_latent_space.csv', index=False)
print("File successfully saved as athlete_latent_space.csv")


#########################################################################################################################
#running isolation forest
latent_full_df = pd.read_csv('athlete_latent_space.csv')


top_10, iso_forest = IF.find_anomalies(latent_full_df, contam = icfg.contam, estimators= icfg.estimators, top = icfg.top) # config file set to 0.002 = this shows the top 10 samples 

print(f"the top 5% most anomalous samples:")
print(top_10[['id', 'scores', 'z1', 'z2', 'z3']])

top_10_IF = pd.DataFrame(top_10, columns=['id', 'scores', 'z1', 'z2', 'z3'])
top_10_IF.to_csv('top_10_IF_flagged', index=False)
