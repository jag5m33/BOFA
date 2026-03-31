import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D # Required for 3D plotting
from tensorflow.keras import models, layers, callbacks

def run_ae(train_data, full_data, cfg):
    """
    Builds and trains an Autoencoder, then extracts the Latent Space.
    """
    input_dim = train_data.shape[1]
    
    # 1. Architecture
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(16, activation='relu')(inputs)
    bottleneck_layer = layers.Dense(cfg.latent_dim, activation='linear', name='bottleneck')(x)
    x = layers.Dense(16, activation='relu')(bottleneck_layer)
    outputs = layers.Dense(input_dim, activation='linear')(x)
    
    autoencoder = models.Model(inputs, outputs)
    encoder = models.Model(inputs, bottleneck_layer) # Extracts the latent coordinates
    
    # 2. Training
    autoencoder.compile(optimizer='adam', loss='mse')
    early_stop = callbacks.EarlyStopping(monitor='loss', patience=cfg.patience, restore_best_weights=True)
    
    autoencoder.fit(train_data, train_data, 
                    epochs=cfg.epochs, 
                    batch_size=cfg.batch_size, 
                    callbacks=[early_stop], 
                    verbose=0)
    
    # 3. Inference
    recons = autoencoder.predict(full_data)
    scores = np.mean(np.square(full_data - recons), axis=1) # Reconstruction Error
    
    latent_train = encoder.predict(train_data)
    latent_full = encoder.predict(full_data)
    
    return scores, latent_train, latent_full

def visualize_latent_3d_gender(latent_full, df, save_path="latent_3d_gender.png"):
    """
    Reduces Latent Space to 3D for visualization, separating by gender
    and highlighting male GH_CONTROL.
    """
    print("Computing 3D t-SNE projection (this can be slow)...")
    tsne = TSNE(n_components=3, perplexity=30, random_state=42)
    z_tsne = tsne.fit_transform(latent_full)
    
    # Create the 3D Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define color palette (Reference vs Doped)
    colors = {'ATHLETE_REF': '#1f77b4', 'GH_CONTROL': '#d62728'}
    
    # Create distinct sets for plotting
    male_indices = df['sex'] == 'Male'
    female_indices = df['sex'] == 'Female'
    
    # Highlight mask: Male & GH_CONTROL
    highlight_mask = (df['sex'] == 'Male') & (df['source'] == 'GH_CONTROL')
    
    # 1. Plot FEMALE points (Base layer)
    ax.scatter(z_tsne[female_indices, 0], 
               z_tsne[female_indices, 1], 
               z_tsne[female_indices, 2], 
               c=df[female_indices]['source'].map(colors),
               alpha=0.3, s=40, label='Female Group', marker='^')
    
    # 2. Plot MALE points (excluding highlight group)
    non_highlight_male = male_indices & ~highlight_mask
    ax.scatter(z_tsne[non_highlight_male, 0], 
               z_tsne[non_highlight_male, 1], 
               z_tsne[non_highlight_male, 2], 
               c=df[non_highlight_male]['source'].map(colors),
               alpha=0.4, s=50, label='Male Group', marker='o')
    
    # 3. Plot MALE GH_CONTROL (Highlight)
    ax.scatter(z_tsne[highlight_mask, 0], 
               z_tsne[highlight_mask, 1], 
               z_tsne[highlight_mask, 2], 
               facecolors='none', edgecolors='#9467bd', 
               s=180, label='Male GH_CONTROL (Highlight)', marker='o', lw=2)

    # Styling the 3D plot
    ax.set_title('3D t-SNE Latent Space: Gender Split & GH_CONTROL Highlight', fontweight='bold', fontsize=14)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_zlabel('t-SNE 3')
    
    # Handle Legend: Remove duplicate labels if necessary and place off-plot
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')

    # View angle (optional: adjust to rotate the final image)
    ax.view_init(elev=20., azim=45) 
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    plt.show()