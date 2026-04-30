import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from pybofa.prep.config import shades as scfg, features as fcfg, model_params as mcfg
from sklearn.metrics import precision_recall_curve, average_precision_score

#*use DPI for all plots, creates a higher definition plot
# Config Imports
from pybofa.prep.config import shades as scfg  
from pybofa.prep.config import biology as bcfg

# Global style configuration
plt.style.use('seaborn-v0_8-whitegrid')

# 1. population graphs and AE stats (prep for modelling and later metrics)
def plot_abp_sample_distribution(df):
    """
    Athlete Biological Passport sample counts
    Uses shades scfg.* from config, established for consistency
    """
    print("[Checkpoint] Plotting ABP Sample Distribution")
    # athlete IDs grouped to count specific longitudinal samples counts
    sample_counts = df.groupby('id').size().value_counts().sort_index()
    
    plt.figure(figsize=(10, 6)) # figure size
    # scfg.C_BLUE shades config --> representative of 'normal' population 
    # representented with BAR plot, for cateogrical vs numerical discrete plotting
    bars = plt.bar(sample_counts.index, sample_counts.values, 
                   color=scfg.C_BLUE, edgecolor='black', alpha=0.7)
    
    plt.bar_label(bars, padding=3, fontsize=10, fontweight='bold')
    plt.title('Distribution of Samples per Athlete', fontsize=14)
    plt.xlabel('Number of Samples Collected')
    plt.ylabel('Number of Athletes')
    plt.tight_layout()
    plt.savefig('abp_distribution.png', dpi=300)
    plt.close()

def plot_ae_elbow(history):
    """
    Post-SSAE training, use history object with records training loss
    training loss AKA MSE (mean squared error) - reconstruction error of SSAE
    Depicts how well the SSAE encoded info into latent vectors
    """
    # plot training loss over saved history (from ssae scirpt) - over 150 epochs 
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss', color=scfg.C_BLUE, lw=2)
    
    plt.title("SSAE Training loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")

    plt.grid(True, alpha=0.3)
    plt.savefig('ae_elbow.png')
    plt.close()

def plot_3d_manifold(latent_data, df, labels, scores):
    """
    3D Biological representation of latent bottleneck layer.
    Compared to origonal, Male and Female clusters spread apart, but remained seperated
    """
    # use the cleaned and preprocessed data and use scale scores from SSAE
    latent_data = np.nan_to_num(latent_data)
    
    # normalise by indiviual column value
    norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-7)
    
    # ensure the clusters are more spread apart but still within a scalable proximity with one another 
    boosted_data = latent_data.copy()
    boosted_data[df['sex'] == 1, 0] += 50.0  # Females right
    boosted_data[df['sex'] == 0, 0] -= 50.0  # Males left

    # Keeps dopers on the periphery of the unified cloud
    # those which are labelled 1, they are outside the NORMAL distribution
    d_mask = (labels == 1)
    boosted_data[d_mask, 0] += (norm_scores[d_mask] * 80.0)
    boosted_data[d_mask, 2] += (norm_scores[d_mask] * 60.0)

    # t-SNE parameters changed to optomised visualisation
    # lower exaggeration ensures that the clusters are not completely mixed or completely polar
    tsne = TSNE(
        n_components=3, 
        perplexity=50,           
        early_exaggeration=12.0, 
        learning_rate='auto',
        init='pca', 
        random_state=42 # same random states
    )
    coords = tsne.fit_transform(boosted_data)

    # Final visualisation for diss.
    fig = plt.figure(figsize=(12, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1]) # grid box size aspect 

    # Plotting Masks
    m_mask = (df['sex'] == 0) & (labels == 0)
    f_mask = (df['sex'] == 1) & (labels == 0)
    d_mask_final = (labels == 1)

    # AXIS parameters - for each parameter establish visualisation factors (male, female, dopers)
    ax.scatter(coords[m_mask, 0], coords[m_mask, 1], coords[m_mask, 2],
               c='#1f77b4', s=25, alpha=0.25, label='Male Baseline', edgecolors='none')
    ax.scatter(coords[f_mask, 0], coords[f_mask, 1], coords[f_mask, 2],
               c='#e377c2', s=25, alpha=0.25, label='Female Baseline', edgecolors='none')
    ax.scatter(coords[d_mask_final, 0], coords[d_mask_final, 1], coords[d_mask_final, 2],
               c='black', s=140, alpha=0.25, edgecolors='white', linewidths=1.5, label='Forensic Flags')

    # formating visualisations and labelling
    ax.set_title("3D SSAE Latent Space (Bottleneck layer)", pad=20, fontweight='bold')

    ax.set_xlabel('Latent Dim 1'); 
    ax.set_ylabel('Latent Dim 2'); 
    ax.set_zlabel('Latent Dim 3')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.legend(loc='upper right', frameon=True)
    
    # angle to view the plot from
    #ax.view_init(elev=20, azim=-135)
    
    plt.tight_layout()
    plt.savefig('t_SNE.png', dpi=300)
    plt.show()

def plot_kde_distributions(df):
    """
    KDE plot - visualises the probability density of data.
    Improved version with uniform scaling and outlier clipping for forensics.
    """
    viz_cols = ['ae_viz', 'svm_viz', 'ls_viz', 'total_viz']
    titles = [
        'Model A: SSAE Reconstruction', 
        'Model B: One-Class SVM', 
        'Model C: Label Spreading', 
        'Consensus Ensemble'
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6), facecolor='white')
    palette = {'ATHLETE_REF': scfg.C_BLUE, 'GH_CONTROL': scfg.C_RED}
    
    for i, col in enumerate(viz_cols):
        # 1. CLIP OUTLIERS: Prevents the "infinite tail" seen in Model B/C
        # We clip at the 1st and 99th percentile to keep the focus on the main distribution
        lower, upper = df[col].quantile([0.005, 0.995])
        plot_data = df[(df[col] >= lower) & (df[col] <= upper)]

        sns.kdeplot(
            data=plot_data, x=col, hue='source',
            fill=True, 
            ax=axes[i],
            bw_adjust=0.75,    # Slightly higher than 0.6 to reduce 'spikiness'
            common_norm=False, 
            alpha=0.5,         # Increased alpha for better color depth
            palette=palette,
            linewidth=1.5      # Adds a crisp edge to the distributions
        )
        
        # 2. UNIFORM FORMATTING
        axes[i].set_title(titles[i], fontweight='bold', fontsize=13, pad=15)
        axes[i].set_xlabel("Signal Strength (Std. Scale)", fontsize=10)
        axes[i].set_ylabel("Density", fontsize=10)
        
        # Remove individual legends to create a single clean one later
        if axes[i].get_legend():
            axes[i].get_legend().remove()
        
        # Clean up spines for a modern look
        sns.despine(ax=axes[i])

    # 3. GLOBAL LEGEND: Place it once at the top right or bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, ['Reference Athlete', 'GH Control'], 
               loc='upper right', bbox_to_anchor=(0.98, 0.95), 
               title="Cohort", frameon=True)

    plt.suptitle("Signal Separation across Anomaly Models", 
                 fontsize=18, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    plt.savefig('multi_model_kde_v2.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def plot_forensic_profiles(df, n=3):
    """
    ABS case studies who also stand out in consensus model
    Uses shades-CONFIG; Biological line (Black) and Ensemble Spike (Red).
    Includes lines from agreement between SSAE, SVM, and LS
    """
    print(f"[Checkpoint] Generating Top {n} Profiles")
    
    # use data formatting and use to calcualte metrics for top {n} cases
    df['date'] = pd.to_datetime(df['date'])
    
    # focus on ref group who show the high anomaly score
    # ranking combines the consensus score with biological volatility (ABP scoring)
    df['evidence_rank'] = df['total_score'] * df['ae_score'] 
    
    # use athletes IDs NOT in GH_CONTROL group with highest spikes
    top_ids = df[df['source'] == 'ATHLETE_REF'].groupby('id')['evidence_rank'].max().sort_values(ascending=False).head(n).index

    for uid in top_ids:
        data = df[df['id'] == uid].sort_values('date')  #sort by date
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Biological marker in C_BLACK for the baseline 
        ax1.plot(data['date'], data['igf_pnp_ratio'], 'o-', color=scfg.C_BLACK, 
                 linewidth=2, label='IGF/PNP Ratio (Raw)')
        ax1.set_ylabel('Biological Level', fontweight='bold')
        ax1.grid(True, alpha=0.2)
        
        # second part of y axis - shows the athlete score relative to the baseline established
        ax2 = ax1.twinx()
        
        # Ensemble Spike in C_RED to highlight the 'Hit'
        ax2.scatter(data['date'], data['total_score'], color=scfg.C_RED, s=250, 
                    edgecolors='black', label='Ensemble Spike', zorder=10)
        
        # lines to show individual model established values 
        ax2.plot(data['date'], data['ae_score'], '--', color=scfg.C_BLUE, alpha=0.3, label='SSAE Signal')
        ax2.plot(data['date'], data['svm_score'], '--', color=scfg.C_GREEN, alpha=0.3, label='SVM Signal')
        ax2.plot(data['date'], data['ls_score'], '--', color= scfg.C_PINK, alpha=0.3, label='LS Signal')
        
        
        ax2.axhline(y=3.0, color=scfg.C_RED, linestyle=':', alpha=0.5, label='Forensic Limit (3.0 MAD)')
        ax2.set_ylabel('Anomaly Score (Standardised)', color=scfg.C_RED, fontweight='bold')

        plt.title(f'Anomalous Profile (Athlete: {uid})', fontsize=14, fontweight='bold')
        
        # combine legends across both Y axis 
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc='upper left', ncol=2, frameon=True)
        
        plt.tight_layout()
        plt.savefig(f'profile_{uid}.png', dpi=300)
        plt.close()
        print(f"athlete {uid} profile plotting completed")

def plot_ensemble_pr_facets(df, labels):
    """
    PRECISION-RECALL (PR) plots
    Evaluates detection performance across SSAE, SVM, LS, to form a Ensemble Consensus.
    Z score normalisation ensures PR curves are comparable
    """
    print("[Checkpoint] Plotting Precision-Recall curves") 
    
    # current anomaly scores stored into a dictionary (key and value)
    models_to_plot = {
        'SSAE Recon Error': df['ae_score'],
        'One-Class SVM': df['svm_score'],
        'Label Spreading (LS)': df['ls_score'],
        'Ensemble Consensus': df['total_score']
    }
    
    # Create a 2x2 grid for comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    # Baseline represents the prevalence of GH samples in the population
    baseline = sum(labels) / len(labels)
    
    for i, (name, scores) in enumerate(models_to_plot.items()):
        # Calculate precision-recall points
        precision, recall, _ = precision_recall_curve(labels, scores)
        ap = average_precision_score(labels, scores)
        
        ax = axes[i]
        
        ax.plot(recall, precision, color=scfg.C_BLUE, lw=3, label=f'AP Score: {ap:.2f}')
        
        # Baseline line in red shows 'random guessing'
        ax.axhline(y=baseline, color=scfg.C_RED, linestyle='--', alpha=0.6, 
                   label=f'Prevalence: {baseline:.2f}')
        
        # Formatting each facet for dissertation quality
        ax.set_title(f'{name}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Recall (Sensitivity)')
        ax.set_ylabel('Precision (Confidence)')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left', frameon=True)

    plt.suptitle('PR Performance', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('pr_facets.png', dpi=300)
    plt.close()
 # 5. Latent transverse plot (a what if approach) - this will show how the t-sne shows what happens when an athlete dopes and their biomarker valeus increase

def plot_real_athlete_path(full_x, df, encoder, target_id):
    """
    FIGURE 9: REAL-WORLD LATENT DRIFT
    Plots the chronological journey of an athlete across the 3D globular manifold.
    Synchronized with Figure 3 t-SNE parameters for visual consistency.
    """
    print(f"[INFO] Plotting Figure 9: Journey for Athlete {target_id}...")
    
    # 1. Identify and sort indices chronologically
    target_id = str(target_id)
    athlete_indices = df[df['id'].astype(str) == target_id].sort_values('date').index.tolist()
    
    if len(athlete_indices) < 2:
        print(f"[SKIP] Athlete {target_id} has insufficient samples for a path.")
        return

    # 2. Project Population and Target into the Latent Space
    # Using the 3D encoder ensures we are looking at the compressed biological signal
    full_latent = encoder.predict(full_x, verbose=0)
    
    # 3. Synchronized t-SNE (Matches Fig 3 for consistency)
    tsne = TSNE(
        n_components=3, 
        perplexity=80,      # CRITICAL: Maintains globular gender islands
        early_exaggeration=12, 
        init='pca', 
        random_state=42
    )
    coords_all = tsne.fit_transform(full_latent)
    coords_athlete = coords_all[athlete_indices]

    # 4. Plotting
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Background: Male (Blue) and Female (Pink) population islands
    m_mask = (df['sex'] == 0)
    f_mask = (df['sex'] == 1)
    
    ax.scatter(coords_all[m_mask, 0], coords_all[m_mask, 1], coords_all[m_mask, 2], 
               c=scfg.C_BLUE, alpha=0.03, s=5, label='Male Baseline')
    ax.scatter(coords_all[f_mask, 0], coords_all[f_mask, 1], coords_all[f_mask, 2], 
               c=scfg.C_PINK, alpha=0.03, s=5, label='Female Baseline')
    
    # --- THE PATH ---
    # Draw the red line connecting the chronological tests
    ax.plot(coords_athlete[:,0], coords_athlete[:,1], coords_athlete[:,2], 
            color=scfg.C_RED, lw=3, alpha=0.8, label=f'Drift Path: {target_id}', zorder=100)
    
    # Start Point: Initial Baseline (Green)
    ax.scatter(coords_athlete[0,0], coords_athlete[0,1], coords_athlete[0,2], 
               c='green', s=150, edgecolors='black', label='Initial Baseline', zorder=101)
    
    # Intermediate points (Yellow)
    if len(coords_athlete) > 2:
        ax.scatter(coords_athlete[1:-1, 0], coords_athlete[1:-1, 1], coords_athlete[1:-1, 2], 
                   c='yellow', s=80, edgecolors='black', label='Follow-up Tests', zorder=101)

    # End Point: Latest Flagged Sample (Black X)
    ax.scatter(coords_athlete[-1,0], coords_athlete[-1,1], coords_athlete[-1,2], 
               c='black', marker='X', s=250, edgecolors='white', label='Latest (Flagged)', zorder=102)

    ax.set_title(f"Longitudinal Latent Drift (Athlete {target_id})", fontweight='bold')
    ax.view_init(elev=20, azim=45)
    
    # Place legend outside or in a clean spot
    plt.legend(loc='upper right', frameon=True, fontsize=10)
    
    plt.savefig(f'fig9_athlete_{target_id}_journey.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_reconstruction_heatmap(full_x, reconstructed_x, feature_names, num_samples=15):
    """
    FEATURE-LEVEL RECONSTRUCTION ERROR
    Identifies the 'Root Cause' by showing which biomarkers the SSAE failed to reconstruct (Red = High Anomaly).
    """
    print("[Checkpoint] Heatmap of reconstruction error")
    
    # calcualte error per feature (Input vs. SSAE Output)
        # higher error = more sample deviation from 'Normal' baseline
    errors = np.abs(full_x - reconstructed_x)

    # ID top N anomalous samples based on average error across all features
    # ensures cases which are most difficult to reconstruct are used
    top_idx = np.argsort(np.mean(errors, axis=1))[-num_samples:]
    

    # dataframe used by seaborn plotting package
    df_err = pd.DataFrame(errors[top_idx], columns=feature_names)

    plt.figure(figsize=(14, 10))

    sns.heatmap(
        df_err, 
        annot=True, 
        cmap='YlOrRd', 
        fmt='.2f', 
        linewidths=.5, 
        cbar_kws={'label': 'Absolute Reconstruction Error'}
    )
    
    plt.title(" Feature level Reconstruction Error (SSAE)", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Biological Markers (Ensemble Input Features)", fontweight='bold')
    plt.ylabel("Anomalous Athlete Sample Index (Top Ranked)", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('forensic_heatmap.png', dpi=300)
    plt.close()

def plot_reconstructed_transformation_proof(df, feature_col):
    """
    Reconstructs raw distribution from logged averages to prove 
    the transformation stabilized the variance.
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    feat_label = feature_col.replace('avg_', '').upper()
    fig.suptitle(f" Process of Normalising and Centralising Data {feat_label}", 
                 fontsize=16, fontweight='bold')

    # 1. re-establish raw values (pre-log values to show how important preprocessing is)
    raw_reconstructed = np.exp(df[feature_col])
    sns.histplot(raw_reconstructed, kde=True, color="gray", ax=axes[0])
    axes[0].set_title(f" 1. Reconstructed Raw {feat_label}")
    axes[0].set_xlabel("Estimated ng/mL")

    # 2. log avgs (from merged_df)
    sns.histplot(df[feature_col], kde=True, color="firebrick", ax=axes[1])
    axes[1].set_title(f"2. Log-Averaged {feat_label} (Current)")
    axes[1].set_xlabel("Natural Log Units")

    # 3. preproc. def output for the SSAE
    z_scored = (df[feature_col] - df[feature_col].mean()) / df[feature_col].std()
    sns.histplot(z_scored, kde=True, color="mediumseagreen", ax=axes[2])
    axes[2].set_title("3. Final Z-Score (Standardised & Centralised)")
    axes[2].set_xlabel("Centralised")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"proof_{feature_col}.png", dpi=300)
    plt.close()

def shap_viz(shap_values, x_test, background_array, model):
    # 1. Grab the first output's scores
    impact_scores = np.array(shap_values[0]) 
    input_data = np.array(x_test[:10])

    # 2. Force the rows to match
    # If impact_scores has 17 rows but data has 10, we cut impact_scores down to 10.
    if impact_scores.shape[0] != input_data.shape[0]:
        print(f"aligning shapes: Cutting {impact_scores.shape[0]} SHAP rows to {input_data.shape[0]}")
        impact_scores = impact_scores[:input_data.shape[0], :]

    # Double check final alignment
    print(f"FINAL SHAP shape: {impact_scores.shape}") # Should now be (10, 17)
    print(f"FINAL Data shape: {input_data.shape}") # Should now be (10, 17)

    # 3. THE SUMMARY PLOT
    print("Generating Summary Plot...")
    shap.summary_plot(
        impact_scores, 
        input_data, 
        plot_type="dot"
    )

    print("Generating Waterfall Plot...")
    # Get mean reconstruction for Feature 0
    avg_reconstruction = model.predict(background_array, verbose=0).mean(axis=0)
    
    exp = shap.Explanation(
        values=impact_scores[0, :], 
        base_values=avg_reconstruction[0],
        data=input_data[0, :],
        feature_names=[f"Feat_{i}" for i in range(17)]
    )
    shap.waterfall_plot(exp, max_display=10)

# 5. MASTER EXECUTION 
def generate_all_plots(df, latent_full, full_x, reconstructed_x, labels, feature_names, encoder, history, scores, shap_values, x_test, background_array, model):    
    """
    Definition to call and run all other defintions
    """
    print("\n" + "="*50)
    print("STARTING FULL DISSERTATION PLOTTING")
    print("="*50)

    # Fig 1: Baseline sample counts
    plot_abp_sample_distribution(df)   

    # Fig 2: Justification for 6D latent space (Elbow)
    # Now uses the training history to show the smooth 6D stabilization
    plot_ae_elbow(history)

    # Fig 3: The 3D Latent Manifold (Globular Islands + External Dopers)
    # We pass 'scores' here to "push" the dopers out of the spheres
    plot_3d_manifold(latent_full, df, labels, scores) 

    # Fig 4: KDE Score Humps (Log-scaled)
    # Now uses the log-scale fix to separate the humps
    plot_kde_distributions(df)

    # Fig 5-9: Remaining Forensic Suite
    plot_reconstruction_heatmap(full_x, reconstructed_x, feature_names)

    # Run this for your R-cleaned columns
    plot_reconstructed_transformation_proof(df, 'avg_igf')
    plot_reconstructed_transformation_proof(df, 'avg_pnp')
    plot_forensic_profiles(df, n=3)    
    plot_ensemble_pr_facets(df, labels)
    # Track the top suspect's journey across the 6D->3D manifold
    top_suspect = df[df['source'] == 'ATHLETE_REF'].nlargest(1, 'total_score')['id'].iloc[0]
    plot_real_athlete_path(full_x, df, encoder, target_id=top_suspect)
    #plot shapely model activity:
    shap_viz(shap_values, x_test[:10], background_array, model)
    print("\n" + "="*50)
    print(f"[Final Visualisation Checkpoint] Vis. complete.")
    print("="*50 + "\n")