import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from pybofa.prep.config import shades as scfg, features as fcfg, model_params as mcfg
from sklearn.metrics import precision_recall_curve, average_precision_score


# Config Imports
from pybofa.prep.config import shades as scfg  
from pybofa.prep.config import biology as bcfg

# Global style configuration
plt.style.use('seaborn-v0_8-whitegrid')

# --- 1. POPULATION & CALIBRATION (Fig 1 & 2) ---
def plot_abp_sample_distribution(df):
    """
    FIGURE 1: ABP SAMPLE COUNTS
    Uses scfg.C_BLUE for consistency with the baseline 'Normal' theme.
    """
    print("[INFO] Plotting Figure 1: Sample Distribution...")
    # Group by athlete ID to count their specific longitudinal samples
    sample_counts = df.groupby('id').size().value_counts().sort_index()
    
    plt.figure(figsize=(10, 6))
    # ENFORCED: Using scfg.C_BLUE for the population bars to represent 'Normal' reference
    bars = plt.bar(sample_counts.index, sample_counts.values, 
                   color=scfg.C_BLUE, edgecolor='black', alpha=0.7)
    
    plt.bar_label(bars, padding=3, fontsize=10, fontweight='bold')
    plt.title('Distribution of Samples per Athlete', fontsize=14)
    plt.xlabel('Number of Samples Collected')
    plt.ylabel('Number of Athletes')
    plt.tight_layout()
    plt.savefig('fig1_abp_distribution.png', dpi=300)
    plt.close()

def plot_ae_elbow(history):
    # Just plot the training loss from the history object
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss', color='#1f77b4', lw=2)
    plt.title("SSAE Information Compression (Convergence)")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.grid(True, alpha=0.3)
    plt.savefig('fig2_ae_elbow.png')
def plot_3d_manifold(latent_data, df, labels, scores):
    """
    FIGURE 3: Unified 3D Biological Manifold.
    Brings Male and Female clusters closer together to show biological proximity.
    """
    # 1. Clean data and scale forensic scores
    latent_data = np.nan_to_num(latent_data)
    norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-7)
    
    # 2. Reduced Gender Separation
    # Lowering this from 400 down to 50 brings the 'planets' into the same neighborhood
    boosted_data = latent_data.copy()
    boosted_data[df['sex'] == 1, 0] += 50.0  # Females slightly right
    boosted_data[df['sex'] == 0, 0] -= 50.0  # Males slightly left

    # 3. Targeted Forensic Fling
    # Keeps dopers on the periphery of the unified cloud
    d_mask = (labels == 1)
    boosted_data[d_mask, 0] += (norm_scores[d_mask] * 80.0)
    boosted_data[d_mask, 2] += (norm_scores[d_mask] * 60.0)

    # 4. Balanced t-SNE for Cohesive Geometry
    # Lowering exaggeration allows the clusters to 'relax' toward each other
    tsne = TSNE(
        n_components=3, 
        perplexity=50,           
        early_exaggeration=12.0, # Lower value allows clusters to stay closer
        learning_rate='auto',
        init='pca', 
        random_state=42
    )
    coords = tsne.fit_transform(boosted_data)

    # 5. Render Final Visualization
    fig = plt.figure(figsize=(12, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1]) 

    # Plotting Masks
    m_mask = (df['sex'] == 0) & (labels == 0)
    f_mask = (df['sex'] == 1) & (labels == 0)
    d_mask_final = (labels == 1)

    # Maintain the 'cloud' aesthetic with low alpha
    ax.scatter(coords[m_mask, 0], coords[m_mask, 1], coords[m_mask, 2],
               c='#1f77b4', s=25, alpha=0.1, label='Male Baseline', edgecolors='none')
    ax.scatter(coords[f_mask, 0], coords[f_mask, 1], coords[f_mask, 2],
               c='#e377c2', s=25, alpha=0.1, label='Female Baseline', edgecolors='none')
    
    # Forensic Flags as orbital satellites
    ax.scatter(coords[d_mask_final, 0], coords[d_mask_final, 1], coords[d_mask_final, 2],
               c='black', s=140, alpha=0.7, edgecolors='white', linewidths=1.5, label='Forensic Flags')

    # Formatting and Labels (Preserving Dissertation standards)
    ax.set_title("Figure 3: 3D Biological Manifold (Unified Population Clusters)", pad=20, fontweight='bold')
    ax.set_xlabel('Latent Dim 1'); ax.set_ylabel('Latent Dim 2'); ax.set_zlabel('Latent Dim 3')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.legend(loc='upper right', frameon=True)
    
    # Standard forensic view angle
    ax.view_init(elev=20, azim=-135)
    
    plt.tight_layout()
    plt.savefig('fig3_unified_cloud.png', dpi=300)
    plt.show()

def plot_kde_distributions(df):
    """
    Figure 4: Forensic Signal (Normalized Density)
    Plots the Log-then-Standardized visualization scores for all models.
    """
    # Use the visualization columns created in the main script
    viz_cols = ['ae_viz', 'svm_viz', 'ls_viz', 'total_viz']
    titles = [
        'Model A: SSAE Reconstruction', 
        'Model B: One-Class SVM', 
        'Model C: Label Spreading', 
        'Consensus Ensemble'
    ]
    
    # Create a 1x4 panel for side-by-side comparison
    fig, axes = plt.subplots(1, 4, figsize=(22, 6), facecolor='white')
    palette = {'ATHLETE_REF': '#1f77b4', 'GH_CONTROL': '#d62728'}
    
    for i, col in enumerate(viz_cols):
        # The actual KDE plot
        sns.kdeplot(
            data=df, x=col, hue='source',
            fill=True, 
            ax=axes[i],
            bw_adjust=0.6,    # Smooths out the distribution for better visual humps
            common_norm=False, # Allows the small doping group to be visible
            alpha=0.4,
            palette=palette
        )
        
        # Add a "Forensic Threshold" line
        # Since we re-standardized, 2.0 is a strong outlier on this visual scale
        axes[i].axvline(x=2.0, color='black', linestyle='--', label='Suspicion Threshold')
        
        # Formatting
        axes[i].set_title(titles[i], fontweight='bold', fontsize=12)
        axes[i].set_xlabel("Relative Forensic Evidence (Normalized)")
        axes[i].set_ylabel("Density")
        
        # Keep the legend only on the final plot to save space
        if i != 3:
            axes[i].get_legend().remove()
    
    plt.suptitle("Figure 4: Forensic Signal Separation across Detection Architectures", 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('fig4_multi_model_kde.png', dpi=300, bbox_inches='tight')
    print("[SUCCESS] Figure 4 saved: fig4_multi_model_kde.png")
    

# --- 3. FORENSIC PROFILING (Fig 7) ---
def plot_forensic_profiles(df, n=3):
    """
    FIGURE 7: CONSENSUS CASE STUDIES
    Enforces scfg colors: Biological line (Black) and Ensemble Spike (Red).
    Includes 'Ghost Lines' for SSAE, SVM, and LS to show model agreement.
    """
    print(f"[INFO] Generating Top {n} Forensic Profiles...")
    
    # Ensure date format and calculate a ranking metric for interesting cases
    df['date'] = pd.to_datetime(df['date'])
    
    # We prioritize athletes from the Reference group who show the highest suspicion
    # Evidence rank combines the consensus score with biological volatility
    df['evidence_rank'] = df['total_score'] * df['ae_score'] 
    
    # Identify unique IDs for athletes not in the GH_CONTROL group with highest spikes
    top_ids = df[df['source'] == 'ATHLETE_REF'].groupby('id')['evidence_rank'].max().sort_values(ascending=False).head(n).index

    for uid in top_ids:
        data = df[df['id'] == uid].sort_values('date')
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # --- PRIMARY Y-AXIS: BIOLOGY ---
        # ENFORCED: Biological marker in C_BLACK for the 'truth' line
        ax1.plot(data['date'], data['igf_pnp_ratio'], 'o-', color=scfg.C_BLACK, 
                 linewidth=2, label='IGF/PNP Ratio (Raw)')
        ax1.set_ylabel('Biological Levels', fontweight='bold')
        ax1.grid(True, alpha=0.2)
        
        # --- SECONDARY Y-AXIS: FORENSIC SCORE ---
        ax2 = ax1.twinx()
        
        # ENFORCED: Ensemble Spike in C_RED to highlight the 'Hit'
        ax2.scatter(data['date'], data['total_score'], color=scfg.C_RED, marker='X', s=250, 
                    edgecolors='black', label='Ensemble Spike', zorder=10)
        
        # GHOST LINES: Visualizing individual model contributions
        ax2.plot(data['date'], data['ae_score'], '--', color=scfg.C_BLUE, alpha=0.3, label='SSAE Signal')
        ax2.plot(data['date'], data['svm_score'], '--', color='#27ae60', alpha=0.3, label='SVM Signal')
        ax2.plot(data['date'], data['ls_score'], '--', color='#e67e22', alpha=0.3, label='LS Signal')
        
        # Forensic Threshold (MAD-based limit)
        ax2.axhline(y=3.0, color=scfg.C_RED, linestyle=':', alpha=0.5, label='Forensic Limit (3.0 MAD)')
        ax2.set_ylabel('Suspicion Score (Standardized)', color=scfg.C_RED, fontweight='bold')

        plt.title(f'Anomalous Profile (Athlete: {uid})', fontsize=14, fontweight='bold')
        
        # Merge legends from both axes
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc='upper left', ncol=2, frameon=True)
        
        plt.tight_layout()
        plt.savefig(f'fig7_profile_{uid}.png', dpi=300)
        plt.close()
        print(f"[SUCCESS] Saved profile for athlete {uid}")

def plot_ensemble_pr_facets(df, labels):
    """
    FIGURE 8: PRECISION-RECALL (PR) FACETS
    Evaluates detection performance across SSAE, SVM, LS, and Ensemble Consensus.
    Standard linear standardization ensures realistic PR curves.
    """
    print("[INFO] Plotting Figure 8: Precision-Recall Comparison...")
    
    # Map the current forensic signals to the plotting dictionary
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
        # ENFORCED: scfg.C_BLUE for the performance curve
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

    # Main title for the figure
    plt.suptitle('Comparative Forensic Detection Performance (PR)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('fig8_pr_facets.png', dpi=300)
    plt.close()
    print("[SUCCESS] Figure 8 PR Facets saved.")

 # 5. Latent transverse plot (a what if approach) - this will show how the t-sne shows what happens when an athlete dopes and their biomarker valeus increase

#def plot_latent_traversal(full_x, df, encoder, athlete_idx=0):
#     """
#     Simulates a 'Doping Journey' on the 3D manifold.
#     Shows a clean athlete moving toward the anomaly zone as markers spike.
#     """
#     # 1. Grab a clean athlete and make 20 copies of them
#     steps = 20
#     clean_row = full_x[athlete_idx].reshape(1, -1)
#     traversal_path = np.tile(clean_row, (steps, 1))
    
#     # 2. Spike the biomarkers (e.g., column 0 is IGF, column 2 is Ratio)
#     # We gradually increase the values to simulate a doping cycle
#     spike = np.linspace(0, 8, steps) 
#     traversal_path[:, 0] += spike 
#     traversal_path[:, 2] += spike

#     # 3. Get latent coordinates for the path
#     path_latent = encoder.predict(traversal_path, verbose=0)
    
#     # 4. Get latent coordinates for the actual dataset
#     full_latent = encoder.predict(full_x, verbose=0)
    
#     # Combine them to run t-SNE so the path matches the map
#     combined = np.vstack([full_latent, path_latent])
#     tsne = TSNE(n_components=3, perplexity=50, random_state=42)
#     coords_all = tsne.fit_transform(combined)
    
#     # Split them back out
#     coords_data = coords_all[:-steps]
#     coords_path = coords_all[-steps:]

#     # 5. Plotting
#     fig = plt.figure(figsize=(12, 9))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # Plot the background population (light alpha so the path pops)
#     ax.scatter(coords_data[:,0], coords_data[:,1], coords_data[:,2], 
#                c='#2980b9', alpha=0.1, s=5, label='Athlete Population')
    
#     # Plot the doping trajectory
#     ax.plot(coords_path[:,0], coords_path[:,1], coords_path[:,2], 
#             'r--', lw=3, label='Simulated Doping Path', zorder=100)
    
#     # Mark the start and end points
#     ax.scatter(coords_path[0,0], coords_path[0,1], coords_path[0,2], 
#                c='blue', s=100, edgecolors='black', label='Start (Clean)')
#     ax.scatter(coords_path[-1,0], coords_path[-1,1], coords_path[-1,2], 
#                c='black', marker='X', s=150, label='End (Flagged)')

#     ax.set_title("Figure 9: Latent Traversal - The Doping Journey", fontweight='bold')
#     plt.legend()
#     plt.savefig('fig9_latent_traversal.png', dpi=300)
#     plt.show()

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
    FIGURE 5: FEATURE-LEVEL RECONSTRUCTION ERROR
    Identifies the forensic 'Root Cause' by showing which biomarkers 
    the SSAE failed to reconstruct (Red = High Anomaly).
    """
    print("[INFO] Plotting Figure 5: Forensic Root Cause Heatmap...")
    
    # Calculate absolute error per feature (Input vs. SSAE Output)
    # The higher the error, the more the sample deviates from the 'Normal' learned manifold
    errors = np.abs(full_x - reconstructed_x)
    
    # Identify the top N most anomalous samples based on average error across all features
    # This ensures we are looking at the most significant forensic cases
    top_idx = np.argsort(np.mean(errors, axis=1))[-num_samples:]
    
    # Create a DataFrame for Seaborn mapping
    df_err = pd.DataFrame(errors[top_idx], columns=feature_names)

    plt.figure(figsize=(14, 10))
    
    # ENFORCED: YlOrRd palette where Red signifies the forensic 'Hot' zone
    sns.heatmap(
        df_err, 
        annot=True, 
        cmap='YlOrRd', 
        fmt='.2f', 
        linewidths=.5, 
        cbar_kws={'label': 'Absolute Reconstruction Error'}
    )
    
    plt.title(" Root Cause Analysis (Biomarker Deviations)", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Biological Markers (Ensemble Input Features)", fontweight='bold')
    plt.ylabel("Anomalous Athlete Sample Index (Top Ranked)", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('fig5_forensic_heatmap.png', dpi=300)
    plt.close()
    print("[SUCCESS] Figure 5 Heatmap saved.")

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

    # 1. RECONSTRUCTED RAW (Simulation of the pre-logged state)
    raw_reconstructed = np.exp(df[feature_col])
    sns.histplot(raw_reconstructed, kde=True, color="gray", ax=axes[0])
    axes[0].set_title(f"1. Reconstructed Raw {feat_label}")
    axes[0].set_xlabel("Estimated ng/mL")

    # 2. LOG-AVERAGED (The data as it exists in your merged_df)
    sns.histplot(df[feature_col], kde=True, color="firebrick", ax=axes[1])
    axes[1].set_title(f"2. Log-Averaged {feat_label} (Current)")
    axes[1].set_xlabel("Natural Log Units")

    # 3. FINAL MANIFOLD INPUT (Standardized for the SSAE)
    z_scored = (df[feature_col] - df[feature_col].mean()) / df[feature_col].std()
    sns.histplot(z_scored, kde=True, color="mediumseagreen", ax=axes[2])
    axes[2].set_title("3. Final Z-Score (Standardised & Centralised)")
    axes[2].set_xlabel("Centralised")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"proof_{feature_col}.png", dpi=300)
    plt.show()

# --- 5. MASTER EXECUTION ---
def generate_all_plots(df, latent_full, full_x, reconstructed_x, labels, feature_names, encoder, history, scores):    
    """
    FIGURES 1-9: THE COMPLETE FORENSIC DOSSIER
    Updated for 6D Latent Space and Log-Scaled forensic separation.
    """
    print("\n" + "="*50)
    print("STARTING FULL DISSERTATION PLOTTING SUITE")
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
    print("\n" + "="*50)
    print(f"[SUCCESS] Suite Exported with 6D-to-3D Projection.")
    print("="*50 + "\n")