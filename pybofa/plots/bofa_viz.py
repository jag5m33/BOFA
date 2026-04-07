import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.preprocessing import RobustScaler, QuantileTransformer

# Config Imports (Based on your directory structure)
from pybofa.prep.config import data as dcfg
from pybofa.prep.config import calibration as ccfg
from pybofa.prep.config import shades as scfg

# Global style
plt.style.use('seaborn-v0_8-whitegrid')

# 1. ELBOW PLOT: PCA
def plot_pca_elbow():
    print("[INFO] Plotting PCA Elbow...")
    df = pd.read_csv(dcfg.final_results)
    z = df.filter(like='latent_dim').values
    pca = PCA().fit(np.nan_to_num(z))
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, label='Individual')
    plt.step(range(1, len(pca.explained_variance_ratio_)+1), np.cumsum(pca.explained_variance_ratio_), label='Cumulative')
    plt.title('PCA Scree Plot (Information Density)')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.legend()
    plt.savefig('pca_elbow.png', dpi=300)
    plt.close()

# 2. ELBOW PLOT: SSAE
def plot_ae_elbow(train_x, choice=6):
    print("[INFO] Plotting SSAE Elbow...")
    mse_list = []
    dims = range(1, 11)
    for d in dims:
        inp = layers.Input(shape=(train_x.shape[1],))
        e = layers.Dense(16, activation='relu')(inp)
        b = layers.Dense(d, activation='linear')(e)
        d_layer = layers.Dense(16, activation='relu')(b)
        out = layers.Dense(train_x.shape[1], activation='linear')(d_layer)
        ae = models.Model(inp, out)
        ae.compile(optimizer='adam', loss='mse')
        ae.fit(train_x, train_x, epochs=10, verbose=0)
        mse_list.append(ae.evaluate(train_x, train_x, verbose=0))

    plt.figure(figsize=(10, 5))
    plt.plot(dims, mse_list, 'o-', color=scfg.C_BLUE)
    plt.axvline(choice, color='red', linestyle='--', label=f'Chosen Dim ({choice})')
    plt.title('SSAE Elbow Analysis')
    plt.xlabel('Bottleneck Size')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig('ssae_elbow.png', dpi=300)
    plt.close()

# 3. PERFORMANCE METRICS (ROC/PR)
def plot_performance():
    print("[INFO] Plotting Performance Metrics...")
    df = pd.read_csv(dcfg.final_results)
    y_true = (df['source'] == ccfg.positive_label).astype(int)
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    fpr, tpr, _ = roc_curve(y_true, df['total_score'])
    ax[0].plot(fpr, tpr, color='orange', label=f'AUC: {auc(fpr, tpr):.2f}')
    ax[0].plot([0,1], [0,1], 'k--')
    ax[0].set_title('ROC Curve')
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].legend()

    p, r, _ = precision_recall_curve(y_true, df['total_score'])
    ax[1].plot(r, p, color='green', label='Ensemble PR')
    ax[1].set_title('Precision-Recall Curve')
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')
    ax[1].legend()
    
    plt.savefig('performance_metrics.png', dpi=300)
    plt.close()

# 4. DIAGNOSTIC SEPARATION (2x2 Model Grid - Sex Distinguished)
def plot_diagnostic_separation():
    print("[INFO] Plotting 2x2 Diagnostic Separation...")
    df = pd.read_csv(dcfg.final_results)
    models = ['ae_score', 'if_score', 'gmm_score', 'svm_score']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, m in enumerate(models):
        df['log_score'] = np.log1p(df[m] - df[m].min())
        
        # Plot Males (Blue) and Females (Red) Reference
        sns.kdeplot(data=df[df['source'] != ccfg.positive_label], x='log_score', 
                    hue='sex', fill=True, palette=[scfg.C_BLUE, 'red'], 
                    ax=axes[i], common_norm=False, alpha=0.3)
        
        # Overlay Doped subjects (Black)
        sns.kdeplot(data=df[df['source'] == ccfg.positive_label], x='log_score', 
                    color='black', ax=axes[i], linewidth=2.5, label='Doped')

        axes[i].set_title(f"{m.upper()} Separation (by Sex)")
        axes[i].set_xlabel('Log(Suspicion Score)')

    plt.tight_layout()
    plt.savefig('diagnostic_separation.png', dpi=300)
    plt.close()

# 5. T-SNE (3D - Sex Separated)
def plot_tsne(latent_data, labels, df_metadata):
    """
    Generates a 3D t-SNE visualization of the Latent Space.
    Fixes the TypeError by using native Matplotlib 3D scatter.
    """
    print("[INFO] Computing 3D t-SNE manifold...")
    
    # 1. Initialize t-SNE for 3 dimensions
    tsne = TSNE(n_components=3, perplexity=30, max_iter=1000, random_state=42)
    coords = tsne.fit_transform(latent_data)
    
    # 2. Setup Figure and 3D Axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 3. Separate Reference (Normal) and Doped samples for layering
    # Assuming 'source' or 'sex' defines the groups in your df_metadata
    is_ref = (labels == 0)
    ref_coords = coords[is_ref]
    doped_coords = coords[~is_ref]
    
    # 4. Plot Reference Population (Light Grey/Translucent)
    ax.scatter(
        ref_coords[:, 0], 
        ref_coords[:, 1], 
        ref_coords[:, 2],
        c='lightgrey',
        s=6,
        alpha=0.15,
        label='Reference Population',
        edgecolor=None,
        linewidth=0
    )
    
    # 5. Plot Doped/Target Samples (Bold Colors)
    # Mapping sex or source to specific colors
    ax.scatter(
        doped_coords[:, 0], 
        doped_coords[:, 1], 
        doped_coords[:, 2],
        c='red', # Or use a mapping based on df_metadata
        s=30,
        alpha=0.8,
        label='Anomalous Samples',
        edgecolor='black',
        linewidth=0.5
    )
    
    # 6. Formatting
    ax.set_title("t-SNE Visualization of SSAE Latent Space (3D)")
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.set_zlabel("t-SNE Dim 3")
    ax.legend()
    
    plt.tight_layout()
    plt.show()
# 6. MODEL COMPARISON PR
def plot_model_comparison_pr():
    print("[INFO] Plotting Multi-Model PR Curve...")
    df = pd.read_csv(dcfg.final_results)
    y_true = (df['source'] == ccfg.positive_label).astype(int)
    
    models = {
        'Autoencoder': 'ae_score',
        'Isolation Forest': 'if_score',
        'GMM': 'gmm_score',
        'SVM': 'svm_score',
        'ENSEMBLE': 'total_score'
    }
    
    plt.figure(figsize=(10, 7))
    for name, col in models.items():
        precision, recall, _ = precision_recall_curve(y_true, df[col])
        plt.plot(recall, precision, label=name, lw=3 if name == 'ENSEMBLE' else 1.5)

    plt.title('Detector Performance Comparison (Precision-Recall)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig('model_comparison_pr.png', dpi=300)
    plt.close()

# 7. ABP MAP
def plot_abp_map():
    print("[INFO] Plotting ABP Map...")
    df = pd.read_csv(dcfg.final_results)
    plt.figure(figsize=(12, 8))
    
    # Reference
    sns.scatterplot(data=df[df['source']!=ccfg.positive_label], 
                    x='igf_pnp_ratio', y='total_score', alpha=0.3, color='blue', label='Reference')
    # Doped
    sns.scatterplot(data=df[df['source']==ccfg.positive_label], 
                    x='igf_pnp_ratio', y='total_score', color='black', marker='*', s=200, label='Doped')
    
    plt.axhline(np.percentile(df['total_score'], 95), color='red', linestyle='--')
    plt.title('Forensic Map: Biological Ratio vs Ensemble Score')
    plt.xlabel('IGF-1 / P-III-NP Ratio (Standardized)') 
    plt.ylabel('Ensemble Suspicion Score')
    plt.savefig('abp_map.png', dpi=300)
    plt.close()