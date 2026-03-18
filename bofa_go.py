#if
#gmm
#svm

# put into consensus approach 
#final anomaly flag
from config import model as mcfg
from config import data as dcfg
import pybofa.prep.autoencoder as aut
import matplotlib.pyplot as plt


#GO PYTHON FILE NEEDS TO HAVE THIS PART - IT IS HOW YOU RUN THE DEFINITION
df_tensor, n_features = aut.proc(dcfg.merged_df) # by saying df_tensor and n_features the definition pulls out the returns and put them in their correspnding variable names 

dims, final_loss_values = aut.elbow_plot(n_features, mcfg.dim)
aut.plotting(dims, final_loss_values)
#run the plotting first to prove that 3 is the best number of dimesions

autoencoder = aut.autoenc_build(n_features, 3)
aut.compiler(autoencoder)
history = aut.train_autoencoder(autoencoder, epochs = mcfg.epoch)

# dims - the dimensions you use, and final loss values - is from the loss function
