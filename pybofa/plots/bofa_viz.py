import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from matplotlib.colors import LogNorm

# Config Imports
from pybofa.prep.config import data as dcfg
from pybofa.prep.config import calibration as ccfg
from pybofa.prep.config import shades as scfg

# Set global plotting style for forensic/academic look
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# --------------------------------------------------------------------------------
# 1. MODEL JUSTIFICATION PLOTS (Elbow & Scree)
# --------------------------------------------------------------------------------

def plot_ae_elbow(train_x, choice=6):
    """AE Elbow Analysis: Justifying the choice of latent dimensions."""
    print("[INFO] Running AE Elbow Analysis...")
    mse_list = []
    dims = range(1, 12)
    init = tf.keras.initializers.GlorotUniform(seed=42)
    
    for d in dims:
        input_dim = train_x.shape[1]
        hidden_dim = int((input_dim + d) / 2) + 4
        
        inp = layers.Input(shape=(input_dim,))
        e = layers.Dense(hidden_dim, activation='relu', kernel_initializer=init)(inp)
        b = layers.Dense(d, activation='linear', kernel_initializer=init)(e)
        d_layer = layers.Dense(hidden_dim, activation='relu', kernel_initializer=init)(b)
        out = layers.Dense(input_dim, activation='linear', kernel_initializer=init)(d_layer)
        
        ae = models.Model(inp, out)
        ae.compile(optimizer='adam', loss='mse')
        ae.fit(train_x, train_x, epochs=15, verbose=0, shuffle=False)
        mse_list.append(ae.evaluate(train_x, train_x, verbose=0))

    plt.figure(figsize=(10, 6))
    plt.plot(dims, mse_list, 'o-', color=scfg.C_BLUE, lw=2)
    plt.axvline(choice, color=scfg.C_ORANGE, linestyle='--', label=f'Chosen Dim: {choice}')
    plt.title('Autoencoder Elbow: Reconstruction Error vs. Bottleneck Size', fontweight='bold')
    plt.xlabel('Latent Dimensions')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend(frameon=True)
    plt.savefig(dcfg.final_results.replace('.csv', '_ae_elbow.png'), dpi=300)
    plt.close()

def plot_elbow_justification():
    """PCA Scree Plot: Justifying the Latent Space dimensionality."""
    print("[INFO] Generating PCA Scree Plot...")
    df = pd.read_csv(dcfg.final_results)
    z = df.filter(like='latent_dim').values
    
    pca = PCA().fit(np.nan_to_num(z))
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    dims = range(1, len(cum_var) + 1)
    
    ax.bar(dims, pca.explained_variance_ratio_, alpha=0.5, color=scfg.C_BLUE, label='Individual Variance')
    ax.step(dims, cum_var, where='mid', color=scfg.C_ORANGE, lw=2, label='Cumulative Variance')
    
    if len(cum_var) >= 3:
        target_dim = 3 
        variance_captured = cum_var[target_dim-1]
        ax.annotate(f'3D Projections capture {variance_captured:.1%} variance', 
                    xy=(target_dim, variance_captured), 
                    xytext=(target_dim + 1, variance_captured - 0.2), 
                    arrowprops=dict(facecolor='black', arrowstyle='->'), 
                    fontweight='bold')

    ax.set_title('PCA Scree Plot: Latent Information Density', fontweight='bold', fontsize=14)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.legend(loc='upper right', frameon=True)
    plt.savefig(dcfg.final_results.replace('.csv', '_scree_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

# --------------------------------------------------------------------------------
# 2. PERFORMANCE SUITE (ROC/PR)
# --------------------------------------------------------------------------------

def plot_results():
    """ROC/PR Suite with wide layout and internal legends."""
    print("[INFO] Generating Performance Suite (ROC/PR)...")
    df = pd.read_csv(dcfg.final_results)
    y_true = (df['source'] == ccfg.positive_label).astype(int)
    
    def jitter(series):
        return series + np.random.normal(0, series.std() * 0.005, len(series))

    fig, ax = plt.subplots(1, 3, figsize=(24, 7)) 
    plt.subplots_adjust(wspace=0.3) 
    
    # Subplot 1: ROC
    fpr, tpr, _ = roc_curve(y_true, df['total_score'])
    ax[0].plot(fpr, tpr, color=scfg.C_ORANGE, lw=3, label=f'Ensemble AUC: {auc(fpr, tpr):.2f}')
    ax[0].plot([0,1], [0,1], 'k--', alpha=0.3)
    ax[0].set_title('ROC Performance', fontweight='bold', fontsize=14)
    ax[0].set_xlabel('False Positive Rate (1 - Specificity)') 
    ax[0].set_ylabel('True Positive Rate (Sensitivity)') 
    ax[0].legend(loc='lower right', frameon=True)

    # Subplot 2: PR Individuals
    models = {'AE': ('ae_score', scfg.C_BLUE), 'IF': ('if_score', scfg.C_ORANGE),
              'GMM': ('gmm_score', scfg.C_GREEN), 'SVM': ('svm_score', scfg.C_BLACK)}
    
    for name, (col, color) in models.items():
        p, r, _ = precision_recall_curve(y_true, jitter(df[col]))
        ax[1].plot(r, p, color=color, alpha=0.7, label=name, lw=2)
        
    ax[1].set_title('Precision-Recall: Individual Heads', fontweight='bold', fontsize=14)
    ax[1].set_xlabel('Recall (Sensitivity)') 
    ax[1].set_ylabel('Precision (PPV)') 
    ax[1].legend(loc='upper right', title="Models", frameon=True)

    # Subplot 3: PR Ensemble
    pe, re, _ = precision_recall_curve(y_true, jitter(df['total_score']))
    ax[2].plot(re, pe, color=scfg.C_GREEN, lw=4, label='Final Ensemble')
    ax[2].set_title('Precision-Recall: Aggregated Ensemble', fontweight='bold', fontsize=14)
    ax[2].set_xlabel('Recall (Sensitivity)') 
    ax[2].set_ylabel('Precision (PPV)') 
    ax[2].legend(loc='upper right', frameon=True)
    
    plt.savefig(dcfg.final_results.replace('.csv', '_performance_suite.png'), dpi=300, bbox_inches='tight')
    plt.close()

# --------------------------------------------------------------------------------
# 3. MANIFOLD VISUALIZATION (t-SNE)
# --------------------------------------------------------------------------------

plt.savefig('performance_metrics.png', dpi=300, bbox_inches='tight')

def plot_3d_tsne():
    """
    3D Manifold Visualization.
    FIXED: Added axis labels and optimized sex-specific cluster visibility.
    """
    df = pd.read_csv(dcfg.final_results)
    X = np.nan_to_num(df[[c for c in df.columns if 'latent_dim' in c]].values)

    tsne = TSNE(n_components=3, perplexity=40, init='pca', random_state=42)
    X_3d = tsne.fit_transform(X)

    gh_mask = (df['source'] == ccfg.positive_label)
    m_mask = (df['sex'].isin([0, 'M', 'Male'])) & ~gh_mask
    f_mask = (df['sex'].isin([1, 'F', 'Female'])) & ~gh_mask

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X_3d[m_mask, 0], X_3d[m_mask, 1], X_3d[m_mask, 2],
               c='#3498db', s=10, alpha=0.2, label='Male Reference')
    ax.scatter(X_3d[f_mask, 0], X_3d[f_mask, 1], X_3d[f_mask, 2],
               c='#e67e22', s=10, alpha=0.2, label='Female Reference')
    ax.scatter(X_3d[gh_mask, 0], X_3d[gh_mask, 1], X_3d[gh_mask, 2],
               c='black', s=120, alpha=1.0, label='GH Control (Doped)', 
               edgecolors='white', linewidth=1.5, zorder=100)

    ax.set_xlabel('Latent Feature 1')
    ax.set_ylabel('Latent Feature 2')
    ax.set_zlabel('Latent Feature 3')
    ax.set_title('3D Latent Manifold: SSAE Distribution', fontweight='bold')
    ax.view_init(elev=25, azim=45) 
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('tsne_3d_final.png', dpi=300)
# --------------------------------------------------------------------------------
# 4. DIAGNOSTIC SEPARATION (The "Nudge" Check)
# --------------------------------------------------------------------------------

def plot_ssae_separation(df=None):
    """
    Multi-Head Separation Plot.
    Normalizes all suspicion scores to a 0-1 range for direct forensic comparison.
    """
    print("[INFO] Generating SSAE Multi-Head Separation Plots (Normalized Linear)...")
    if df is None:
        df = pd.read_csv(dcfg.final_results)

    models = ['ae_score', 'if_score', 'gmm_score', 'svm_score']
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    for i, m in enumerate(models):
        # 1. MIN-MAX NORMALIZATION
        # Scales the specific model's scores to a strict 0.0 - 1.0 range
        m_min = df[m].min()
        m_max = df[m].max()
        norm_series = (df[m] - m_min) / (m_max - m_min)
        
        # 2. PLOT
        sns.kdeplot(
            data=df, 
            x=norm_series, 
            hue='source', 
            fill=True, 
            palette=[scfg.C_BLUE, scfg.C_BLACK], 
            ax=axes[i], 
            common_norm=False, 
            alpha=0.5, 
            linewidth=2.5
        ) 
        
        # 3. FORMATTING
        axes[i].set_title(f"Model Head: {m.split('_')[0].upper()}", fontweight='bold', fontsize=15)
        axes[i].set_xlabel('Normalized Suspicion Index (0.0 - 1.0)') 
        axes[i].set_ylabel('Athlete Density') 
        
        # Force identical x-limits for all subplots
        axes[i].set_xlim(-0.05, 1.05)
        
        # Fix Legend labels
        legend = axes[i].get_legend()
        if legend:
            legend.set_title("Population")
            # Ensure order matches your config: 0 is Ref, 1 is Doped
            for t, l in zip(legend.texts, ['Doped (GH)', 'Reference']):
                t.set_text(l)

    plt.suptitle("SSAE Avenue: Normalized Latent Separation Analysis", fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = dcfg.final_results.replace('.csv', '_separation_normalized.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Normalized separation plots saved to: {save_path}")
# --------------------------------------------------------------------------------
# 5. FORENSIC ABP MAP (Fixing "Volatility" logic)
# --------------------------------------------------------------------------------

def plot_abp_suspicion_map():
    """Maps Individual Variance (Volatility) against Ensemble Suspicion."""
    print("[INFO] Generating Fixed ABP Suspicion Map...")
    df = pd.read_csv(dcfg.final_results)
    
    # 1. DYNAMIC COLUMN DETECTION
    # Look for the volatility/variance column created by the ABP script
    possible_cols = ['igf_pnp_ratio_std', 'igf_pnp_ratio_volatility', 'igf_pnp_ratio_cv']
    vol_col = next((c for c in possible_cols if c in df.columns), None)
    
    if vol_col is None:
        print(f"[ERROR] Could not find a variance column. Available columns: {df.columns.tolist()}")
        return

    ae_col = 'total_score'
    
    plt.figure(figsize=(14, 9))
    
    # 2. PLOT REFERENCE POPULATION (By Sex)
    # Filter out the doped samples first
    ref_df = df[df['source'] == ccfg.unlabeled_label]
    
    sns.scatterplot(
        data=ref_df, 
        x=vol_col, 
        y=ae_col,
        hue='sex', 
        palette=[scfg.C_BLUE, scfg.C_ORANGE], 
        alpha=0.3, 
        s=50,
        edgecolor=None
    )
    
    # 3. PLOT DOPED SAMPLES (GH Control)
    doped_df = df[df['source'] == ccfg.positive_label]
    
    sns.scatterplot(
        data=doped_df, 
        x=vol_col, 
        y=ae_col,
        color='black', 
        s=200, 
        label='GH Control (Doped)', 
        marker='*',
        edgecolor='lime', 
        linewidth=1.5, 
        zorder=10
    )

    # 4. FORENSIC THRESHOLDS
    plt.axhline(np.percentile(df[ae_col], 95), color='red', linestyle='--', alpha=0.6, label='95% ML Threshold')
    plt.axvline(np.percentile(df[vol_col].dropna(), 95), color='gray', linestyle=':', alpha=0.6, label='95% Bio-Variance')

    plt.title("Forensic ABP Map: Biological Volatility vs. Ensemble Suspicion", fontweight='bold', fontsize=16)
    plt.xlabel(f"Biological Volatility ({vol_col.replace('_', ' ').title()})", fontsize=12)
    plt.ylabel("Ensemble Suspicion Score (Standardized Z-Score)", fontsize=12)
    
    # Move legend outside so it doesn't cover data points
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=True)
    
    plt.tight_layout()
    plt.savefig(dcfg.final_results.replace('.csv', '_abp_map.png'), dpi=300, bbox_inches='tight')
    plt.close()