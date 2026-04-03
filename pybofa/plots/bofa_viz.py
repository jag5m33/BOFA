import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from matplotlib.colors import LogNorm
from tensorflow.keras import models, layers
from pybofa.prep.config import data as dcfg
from pybofa.prep.config import calibration as ccfg
from pybofa.prep.config import shades as scfg
# color_blind_friendly_palette 

def plot_elbow_justification():
    """PCA Scree Plot with External Legend."""
    df = pd.read_csv(dcfg.latent_full.replace('.npy', '.csv'))
    z = df.filter(like='latent_dim').values
    pca = PCA().fit(np.nan_to_num(z))
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    dims = range(1, len(cum_var) + 1)
    
    ax.bar(dims, pca.explained_variance_ratio_, alpha=0.5, color=scfg.C_BLUE, label='Individual Variance')
    ax.step(dims, cum_var, where='mid', color=scfg.C_ORANGE, lw=2, label='Cumulative Variance')
    
    if len(cum_var) >= 3:
        ax.annotate(f'3D captures {cum_var[2]:.1%} variance', xy=(3, cum_var[2]), 
                    xytext=(5, 0.6), arrowprops=dict(facecolor='black', arrowstyle='->'), fontweight='bold')

    ax.set_title('PCA Scree Plot: Dimensionality Justification', fontweight='bold')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    
    # Place legend outside to the right
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title="Variance Type")
    plt.tight_layout(rect=[0, 0, 0.85, 1.0])
    plt.savefig('scree_plot.png', dpi=300, bbox_inches='tight')

def plot_ae_elbow(train_x, choice):
    """AE Elbow with External Legend."""
    mse_list = []
    dims = range(1, 11)
    # run each dimension through the autoencoder and get the mse for each dimension
    # then plot the mse for each dimension to see the elbow
    for d in dims:
        inp = layers.Input(shape=(train_x.shape[1],))
        e = layers.Dense(16, activation='relu')(inp)
        b = layers.Dense(d, activation='linear')(e)
        d_layer = layers.Dense(16, activation='relu')(b)
        out = layers.Dense(train_x.shape[1], activation='linear')(d_layer)
        ae = models.Model(inp, out); ae.compile(optimizer='adam', loss='mse')
        ae.fit(train_x, train_x, epochs=15, verbose=0)
        mse_list.append(ae.evaluate(train_x, train_x, verbose=0))

    # MSE recorded in MSE list
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dims, mse_list, 'o-', color=scfg.C_BLUE, label='Reconstruction Error (MSE)')
    ax.axvline(choice, color=scfg.C_ORANGE, linestyle='--', label=f'Selected Dimensions ({choice})')
    
    ax.set_title('Autoencoder Elbow Analysis', fontweight='bold')
    ax.set_xlabel('Latent Dimensions')
    ax.set_ylabel('Mean Squared Error')
    
    # Legend outside
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout(rect=[0, 0, 0.82, 1.0]) # draw rectangle on the left to make room for legend
    plt.savefig('ae_elbow.png', dpi=300, bbox_inches='tight')   # dpi - dots per inch - detail down to the pixel
        # bbox_inches = automaticlaly adjusts the bounding box to fit the content, ensuring nothing is cut off (like the legend)
def plot_3d_tsne():
    """Natural 3D Latent Mapping: Optimized for Population Separation."""
    # 1. Load data from the exported results
    df = pd.read_csv(dcfg.final_results)
    
    # 2. Isolate Latent Features for the Manifold
    latent_cols = [c for c in df.columns if 'latent_dim' in c]
    if not latent_cols:
        print("[ERROR] No latent dimensions found in CSV. Check your export step.")
        return
    
    X = df[latent_cols].values

    # 3. HIGH-FIDELITY T-SNE SETTINGS
    # Perplexity 50-60 creates more stable 'global' clusters for large datasets
    # init='pca' ensures the Male/Female split is based on variance, not random noise
    tsne = TSNE(
        n_components=3, 
        perplexity=55, 
        init='pca', 
        learning_rate='auto', 
        n_iter=2000,
        random_state=42
    )
    X_3d = tsne.fit_transform(np.nan_to_num(X))

    # 4. Create Biological Masks
    gh_mask = (df['source'] == ccfg.positive_label)
    m_mask = (df['sex'].isin([0, 'M', 'Male'])) & ~gh_mask
    f_mask = (df['sex'].isin([1, 'F', 'Female'])) & ~gh_mask

    # 5. Visualization
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Reference Groups (Small, transparent dots)
    ax.scatter(X_3d[m_mask, 0], X_3d[m_mask, 1], X_3d[m_mask, 2],
               c=scfg.C_BLUE, s=6, alpha=0.25, label='Male Reference', edgecolors='none')
    
    ax.scatter(X_3d[f_mask, 0], X_3d[f_mask, 1], X_3d[f_mask, 2],
               c=scfg.C_ORANGE, s=6, alpha=0.25, label='Female Reference', edgecolors='none')

    # GH Control (Large, opaque, high-contrast)
    ax.scatter(X_3d[gh_mask, 0], X_3d[gh_mask, 1], X_3d[gh_mask, 2],
               c='black', s=100, alpha=1.0, label='GH Control (Doped)', 
               edgecolors='white', linewidth=1.5, zorder=10)

    # 6. Formatting for Dissertation
    ax.set_title("Forensic Biological Mapping: 3D Latent Space", fontweight='bold', fontsize=16, pad=20)
    
    # Standardize View Angle (Elevated to see depth)
    ax.view_init(elev=25, azim=45)
    
    # Legend Refinement
    leg = ax.legend(loc='upper right', title="Athlete Population", frameon=True, fontsize=11)
    for handle in leg.legend_handles:
        handle._sizes = [80]
        handle.set_alpha(1.0)

    plt.tight_layout()
    plt.savefig('tsne_3d_refined.png', dpi=300)
    print("[SUCCESS] High-fidelity 3D t-SNE saved as tsne_3d_refined.png")

def plot_results():
    """ROC/PR Suite with Legends."""
    df = pd.read_csv(dcfg.final_results)
    y_true = (df['source'] == ccfg.positive_label).astype(int)
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # ROC
    fpr, tpr, _ = roc_curve(y_true, df['total_score'])
    ax[0].plot(fpr, tpr, color=scfg.C_ORANGE, lw=2, label=f'Ensemble AUC: {auc(fpr, tpr):.2f}')
    ax[0].plot([0,1], [0,1], 'k--', alpha=0.3, label='Random Chance')
    ax[0].legend(loc='lower right')
    ax[0].set_title('ROC Performance')

    # PR Individuals
    for m in ['ae_score', 'if_score', 'gmm_score', 'svm_score']:
        p, r, _ = precision_recall_curve(y_true, df[m])
        ax[1].plot(r, p, alpha=0.4, label=m.split('_')[0].upper())
    ax[1].legend(loc='upper right', title="Individual Models", fontsize='small')
    ax[1].set_title('Precision-Recall: Models')

    # PR Ensemble
    pe, re, _ = precision_recall_curve(y_true, df['total_score'])
    ax[2].plot(re, pe, color=scfg.C_GREEN, lw=3, label='Final Ensemble')
    ax[2].legend(loc='upper right')
    ax[2].set_title('Precision-Recall: Ensemble')
    
    plt.tight_layout()
    plt.savefig('performance_metrics.png', dpi=300)

def plot_method_logic():
    """KDE distributions with centered Legend.""" # kernel density estimation - smoothed version of a histogram
        # non-parametric statistical technique used to estimate the density fo random values (view where most points lie per source group per-model output)

    df = pd.read_csv(dcfg.final_results)
    models = ['ae_score', 'if_score', 'gmm_score', 'svm_score']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8)) # facted grid for 4 models, 2 rows and 2 columns
    axes = axes.flatten()

    # count the indexes in models list and iterate through 
    for i, m in enumerate(models):
        # We add +1 because Log cannot handle 0 or negative numbers
        plot_data = df[m] - df[m].min() + 1
        
        sns.kdeplot(data=df, x=plot_data, hue='source', fill=True, 
                    palette=[scfg.C_BLUE, scfg.C_BLACK], ax=axes[i], 
                    common_norm=False, alpha=0.5, log_scale=True) 

    # Create a single legend for the whole figure
    handles = [plt.Rectangle((0,0),1,1, color=scfg.C_BLUE), plt.Rectangle((0,0),1,1, color=scfg.C_BLACK)]
    fig.legend(handles, ['Reference Group', 'GH Control (Doped)'], loc='upper center', ncol=2)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('logic_distributions.png', dpi=300)

def plot_cm(target_recall=0.70):
    """
    Confusion Matrix displaying hits and false alarms at a specific recall target.
    """
    df = pd.read_csv(dcfg.final_results)
    y_true = (df['source'] == ccfg.positive_label).astype(int)
    
    # Calculate threshold based on the target recall of the doped group
    doped_scores = df[df['source'] == ccfg.positive_label]['total_score']
    thresh = np.percentile(doped_scores, (1 - target_recall) * 100)
    y_pred = (df['total_score'] >= thresh).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 7))
    # Using 'Blues' to match your C_BLUE theme
    sns.heatmap(np.log1p(cm), annot = cm, fmt='d', cmap='Blues', cbar=True, norm = LogNorm(),
                xticklabels=['Pred Clean', 'Pred Doped'], 
                yticklabels=['Actual Clean', 'Actual Doped'],
                annot_kws={"size": 14, "weight": "bold"})
    
    plt.title(f'Confusion Matrix (@ {target_recall*100}% Recall Target)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Ground Truth', fontweight='bold')
    plt.xlabel('Algorithm Decision', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print(f"[SUCCESS] Confusion matrix saved at {target_recall*100}% recall.")