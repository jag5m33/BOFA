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
import tensorflow as tf

def plot_elbow_justification():
    """PCA Scree Plot: Justifying the 6D Latent Space."""
    df = pd.read_csv(dcfg.latent_full.replace('.npy', '.csv'))
    # Only look at the latent coordinates created by the AE
    z = df.filter(like='latent_dim').values
    
    pca = PCA().fit(np.nan_to_num(z))
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    dims = range(1, len(cum_var) + 1)
    
    ax.bar(dims, pca.explained_variance_ratio_, alpha=0.5, color=scfg.C_BLUE, label='Individual Variance')
    ax.step(dims, cum_var, where='mid', color=scfg.C_ORANGE, lw=2, label='Cumulative Variance')
    
    # Annotate the 6th Dimension (your choice in config)
    if len(cum_var) >= 6:
        target_dim = 3 
        variance_captured = cum_var[target_dim-1]
        ax.annotate(f'3D Latent Space captures {variance_captured:.1%} variance', 
                    xy=(target_dim, variance_captured), 
                    xytext=(target_dim + 1, variance_captured - 0.2), 
                    arrowprops=dict(facecolor='black', arrowstyle='->'), 
                    fontweight='bold', fontsize=10)

    ax.set_title('PCA Scree Plot: Latent Information Density', fontweight='bold')
    ax.set_xlabel('Principal Component (Latent Dimensions)')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_ylim(0, 1.1) # Give space for the annotation
    
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title="Variance Metrics")
    plt.tight_layout(rect=[0, 0, 0.85, 1.0])
    plt.savefig('scree_plot.png', dpi=300, bbox_inches='tight')

def plot_ae_elbow(train_x, choice):
    """AE Elbow Analysis for latent dimension selection with dynamic architecture."""
    mse_list = []
    dims = range(1, 11)
    
    # Use a fixed seed for the elbow plot so the line doesn't 'wiggle' on redraws
    init = tf.keras.initializers.GlorotUniform(seed=42)
    
    for d in dims:
        input_dim = train_x.shape[1]
        # Dynamic hidden layer: Matches your bofa_go.py logic
        hidden_dim = int((input_dim + d) / 2) + 4
        
        inp = layers.Input(shape=(input_dim,))
        e = layers.Dense(hidden_dim, activation='relu', kernel_initializer=init)(inp)
        b = layers.Dense(d, activation='linear', kernel_initializer=init)(e)
        d_layer = layers.Dense(hidden_dim, activation='relu', kernel_initializer=init)(b)
        out = layers.Dense(input_dim, activation='linear', kernel_initializer=init)(d_layer)
        
        ae = models.Model(inp, out)
        ae.compile(optimizer='adam', loss='mse')
        
        # We use fewer epochs here for speed, but shuffle=False for stability
        ae.fit(train_x, train_x, epochs=20, verbose=0, shuffle=False)
        mse_list.append(ae.evaluate(train_x, train_x, verbose=0))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dims, mse_list, 'o-', color=scfg.C_BLUE, lw=2, label='Reconstruction Error (MSE)')
    ax.axvline(choice, color=scfg.C_ORANGE, linestyle='--', label=f'Selected Dimensions ({choice})')
    
    ax.set_title('Autoencoder Elbow Analysis: Reconstruction Fidelity', fontweight='bold')
    ax.set_xlabel('Latent Dimensions (Bottleneck Size)')
    ax.set_ylabel('Mean Squared Error (MSE)')
    
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout(rect=[0, 0, 0.82, 1.0])
    plt.savefig('ae_elbow.png', dpi=300, bbox_inches='tight')


def plot_3d_tsne():
    df = pd.read_csv(dcfg.final_results)
    X = np.nan_to_num(df[[c for c in df.columns if 'latent_dim' in c]].values)

    # High-Pressure t-SNE: No extra libraries needed
    # We use a massive learning_rate and perplexity to force 'continents'
    tsne = TSNE(
        n_components=3, 
        perplexity=50,           # High value = Global structure focus
        early_exaggeration=15,    # Strongly separates the clusters early on
        init='pca',               # Starts with a stable PCA layout
        learning_rate='auto',       # Pushes points harder to find their groups
        #n_iter=2500,              # More time to settle into clean shapes
        random_state=42
    )
    X_3d = tsne.fit_transform(X)

    gh_mask = (df['source'] == ccfg.positive_label)
    m_mask = (df['sex'].isin([0, 'M', 'Male'])) & ~gh_mask
    f_mask = (df['sex'].isin([1, 'F', 'Female'])) & ~gh_mask

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Styling for depth
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False) 
     
    ax.scatter(X_3d[m_mask, 0], X_3d[m_mask, 1], X_3d[m_mask, 2],
               c=scfg.C_BLUE, s=3, alpha=0.1, label='Male Reference')

    ax.scatter(X_3d[f_mask, 0], X_3d[f_mask, 1], X_3d[f_mask, 2],
               c=scfg.C_ORANGE, s=3, alpha=0.1, label='Female Reference')

    # 2. Doped Group: Solid, large, and outlined in white for forensic contrast
    ax.scatter(X_3d[gh_mask, 0], X_3d[gh_mask, 1], X_3d[gh_mask, 2],
               c='black', s=120, alpha=1.0, label='GH Control (Doped)', 
               edgecolors='white', linewidth=1.5, zorder=100)

    # 3. Clean View: This angle usually separates the Sex-clusters best
    ax.view_init(elev=30, azim=120) 
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Legend formatting
    leg = ax.legend(loc='upper right', title="Athlete Population", frameon=True)
    for handle in leg.legend_handles:
        handle._sizes = [150]
        handle.set_alpha(1.0)

    plt.tight_layout()
    plt.savefig('tsne_3d_refined.png', dpi=300)
    print("[SUCCESS] High-pressure t-SNE generated successfully.")

def plot_results():
    """ROC/PR Suite with clear axis labels on all subplots."""
    df = pd.read_csv(dcfg.final_results)
    y_true = (df['source'] == ccfg.positive_label).astype(int)
    
    def jitter(series):
        return series + np.random.normal(0, series.std() * 0.005, len(series))

    fig, ax = plt.subplots(1, 3, figsize=(20, 6)) 
    
    # 1. ROC Performance
    fpr, tpr, _ = roc_curve(y_true, df['total_score'])
    ax[0].plot(fpr, tpr, color=scfg.C_ORANGE, lw=2, label=f'Ensemble AUC: {auc(fpr, tpr):.2f}')
    ax[0].plot([0,1], [0,1], 'k--', alpha=0.3)
    ax[0].set_title('ROC Performance', fontweight='bold')
    ax[0].set_xlabel('False Positive Rate (1 - Specificity)') # ADDED
    ax[0].set_ylabel('True Positive Rate (Sensitivity)')      # ADDED
    ax[0].legend(loc='lower right')

    # 2. PR Individuals 
    for m in ['ae_score', 'if_score', 'gmm_score', 'svm_score']:
        p, r, _ = precision_recall_curve(y_true, jitter(df[m]))
        ax[1].plot(r, p, alpha=0.5, label=m.split('_')[0].upper())
    ax[1].set_title('Precision-Recall: Models', fontweight='bold')
    ax[1].set_xlabel('Recall (Sensitivity)') # ADDED
    ax[1].set_ylabel('Precision (PPV)')       # ADDED
    ax[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), title="Individual Models")

    # 3. PR Ensemble 
    pe, re, _ = precision_recall_curve(y_true, jitter(df['total_score']))
    ax[2].plot(re, pe, color=scfg.C_GREEN, lw=3, label='Final Ensemble')
    ax[2].set_title('Precision-Recall: Ensemble', fontweight='bold')
    ax[2].set_xlabel('Recall (Sensitivity)') # ADDED
    ax[2].set_ylabel('Precision (PPV)')       # ADDED
    ax[2].legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    
    plt.tight_layout(rect=[0, 0, 0.9, 1.0]) 
    plt.savefig('performance_metrics.png', dpi=300, bbox_inches='tight')


def plot_method_logic():
    """KDE distributions with explicit X/Y labels for each model density."""
    df = pd.read_csv(dcfg.final_results)
    models = ['ae_score', 'if_score', 'gmm_score', 'svm_score']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, m in enumerate(models):
        plot_data = df[m] - df[m].min() + 1
        sns.kdeplot(data=df, x=plot_data, hue='source', fill=True, 
                    palette=[scfg.C_BLUE, scfg.C_BLACK], ax=axes[i], 
                    common_norm=False, alpha=0.5, log_scale=True) 
        
        axes[i].set_title(f"{m.split('_')[0].upper()} Density", fontweight='bold')
        axes[i].set_xlabel('Standardized Suspicion Score (Log Scale)') # ADDED
        axes[i].set_ylabel('Athlete Density')                        # ADDED

    handles = [plt.Rectangle((0,0),1,1, color=scfg.C_BLUE), plt.Rectangle((0,0),1,1, color=scfg.C_BLACK)]
    fig.legend(handles, ['Reference Group', 'GH Control (Doped)'], loc='upper center', ncol=2)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('logic_distributions.png', dpi=300)

def plot_cm(target_recall=0.70):
    """Confusion Matrix with Log-normalized heatmap for forensic transparency."""
    df = pd.read_csv(dcfg.final_results)
    y_true = (df['source'] == ccfg.positive_label).astype(int)
    
    doped_scores = df[df['source'] == ccfg.positive_label]['total_score']
    thresh = np.percentile(doped_scores, (1 - target_recall) * 100)
    y_pred = (df['total_score'] >= thresh).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(np.log1p(cm), annot=cm, fmt='d', cmap='Blues', cbar=True, norm=LogNorm(),
                xticklabels=['Pred Clean', 'Pred Doped'], 
                yticklabels=['Actual Clean', 'Actual Doped'],
                annot_kws={"size": 14, "weight": "bold"})
    
    plt.title(f'Confusion Matrix (@ {target_recall*100}% Recall)', fontsize=14, fontweight='bold')
    plt.ylabel('Ground Truth', fontweight='bold')
    plt.xlabel('Algorithm Decision', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)

def plot_abp_suspicion_map():
    """
    ABP Forensic Map: Matches the longitudinal logic of Nasonov (2024).
    Plots Biological Volatility (CV) vs. Model Detection (AE Score).
    """
    df = pd.read_csv(dcfg.final_results)
    
    # We focus on the most important marker ratio CV from your ABP script
    # Note: Using .fillna(0) for athletes with only 1 sample (no CV possible)
    cv_col = 'igf_pnp_ratio_cv' 
    ae_col = 'ae_score'
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 1. Background: Reference Athletes
    ref_mask = df['source'] == ccfg.unlabeled_label
    ax.scatter(df.loc[ref_mask, cv_col], df.loc[ref_mask, ae_col], 
               c=scfg.C_BLUE, alpha=0.2, s=40, label='Reference Population', edgecolors='none')
    
    # 2. Foreground: Doped Athletes (The 'Targets')
    gh_mask = df['source'] == ccfg.positive_label
    ax.scatter(df.loc[gh_mask, cv_col], df.loc[gh_mask, ae_col], 
               c='black', s=150, alpha=1.0, label='GH Control (Doped)', 
               edgecolors='#00FF00', linewidth=2, zorder=10)

    ax.scatter(df.loc[ref_mask, cv_col], df.loc[ref_mask, ae_col], 
           c=scfg.C_BLUE, alpha=0.5, s=60, label='Reference Population', 
           edgecolors='white', linewidth=0.5, zorder=1)

    # 3. Forensic Thresholds (95th Percentiles)
    ae_thresh = np.percentile(df[ae_col], 95)
    cv_thresh = np.percentile(df[cv_col].dropna(), 95)
    
    ax.axhline(ae_thresh, color='red', linestyle='--', alpha=0.6, label='AE Detection Limit')
    ax.axvline(cv_thresh, color='orange', linestyle='--', alpha=0.6, label='Individual Stability Limit')

    # Shade the 'High Suspicion' Quadrant (Top Right)
    ax.axvspan(cv_thresh, df[cv_col].max(), ae_thresh/df[ae_col].max(), 1, 
               color='red', alpha=0.05, label='High Suspicion Zone')

    # Formatting
    ax.set_title("Forensic ABP Map: Biological Volatility vs. AE Detection", fontweight='bold', fontsize=14)
    ax.set_xlabel("Intra-individual Coefficient of Variation (Volatility %)", fontsize=11)
    ax.set_ylabel("Autoencoder Reconstruction Error (Standardized)", fontsize=11)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig('abp_suspicion_map.png', dpi=300)
    print(f"[SUCCESS] ABP Suspicion Map saved. High-CV threshold: {cv_thresh:.2f}")