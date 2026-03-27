import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pybofa.prep.config import data as dcfg
from pybofa.prep.config import processor as pcfg
import matplotlib.pyplot as plt

# load = preproc. (match autoencoder format for comparisons)

def proc(merged_df_path): 
    
    df = pd.read_csv(merged_df_path)
    
    ids = df['id']
    source = df['source']
    
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
    
    return df_tensor, n_features, ids, source

# build PCA mdoel 
def build_pca(n_components):
    pca = PCA(n_components=n_components)
    return pca

#train on athletes_ref data
def train_pca(pca, data, source):
    athlete_mask = (source.values == 'ATHLETE_REF')
    pca.fit(data[athlete_mask])
    print("PCA trained on athlete_ref only (no leaking of controld ata)")

    return pca

#generate latent space corodinates:

def build_latent_pca_df(latent_data, latent_cols, ids, source, extra_df = None):
    latent_df = pd.DataFrame(latent_data, columns = latent_cols)
    latent_df['id'] = ids.values
    latent_df['source'] = source.values 

    #extra column for gender (sex)
    if extra_df is not None:
        if 'sex' in extra_df.columns:
            latent_df['sex'] = extra_df['sex'].values
            
    return latent_df

def plot_pca_vairance(pca):
    var = pca.explained_variance_ratio_
    cum_var = np.cumsum(var)
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(var)+1), cum_var, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid(True)
    plt.show()

def plot_pca_3d(latent_df):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection = '3d')

    #create points:
    gh_mask = latent_df['source'] == 'GH_CONTROL'
    ath_mask = latent_df['source'] == 'ATHLETE_REF'

    ax.scatter(
        latent_df.loc[ath_mask, 'latent_0'],
        latent_df.loc[ath_mask, 'latent_1'],
        latent_df.loc[ath_mask, 'latent_2'],
        alpha = 0.3,
        label = 'Athlete'
    )
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('3D PCA latent space')
    ax.legend()
    plt.show()

