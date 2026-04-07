import sys 
import os 
project_root = r"c:\Users\jagmeet\vsc\pybofa" 
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pybofa.plots  import bofa_viz as viz    
import pybofa
print(f"DEBUG: Using pybofa from: {pybofa.__file__}")

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.impute import SimpleImputer

# Config & Model Imports
from pybofa.prep.config import data as dcfg
from pybofa.prep.config import model_params as mcfg
from pybofa.prep.config import ensemble_params as ecfg
from pybofa.prep.config import processor as pcfg
from pybofa.prep.config import calibration as ccfg
import pybofa.models.ssae as ssae_mod  
import pybofa.models.IF as if_mod    
import pybofa.models.gmm as gmm_mod 
import pybofa.models.SVM as svm_mod
import pybofa.models.abp as abp

def load_and_preprocess():
    """
    Sex-specific normalization and differentiated anomaly filtering 
    based on biological variance thresholds (M < 30% CV, F < 50% CV).
    """
    print(f"--- Step 1: Loading Data with Sex-Specific Filtering ---")
    df = pd.read_csv(dcfg.merged_df)
    
    # Feature Engineering
    df['igf_pnp_ratio'] = df['avg_igf'] - df['avg_pnp']
    model_cols = pcfg.pnp_cols + pcfg.igf_cols + ['igf_pnp_ratio']
    
    # Pre-calculate CV for filtering 'Normal' training data
    # Note: Using standard deviation proxy if raw CV isn't in merged_df
    #LINES 34 - 50 = 
    df_processed_list = []

    for s in df['sex'].unique(): # loop that runs once for each sex
        gender_group = df[df['sex'] == s].copy()
        
        # BIOLOGICAL THRESHOLDING
        # Defining 'Normal' reference population based on sex-specific variance
        cv_limit = 0.30 if s in [0, 'M', 'Male'] else 0.50  # sets a volatility threshold (males = 30%, females = 50%)

        
        # train_mask identifies the 'True Normal' reference population
        # (Must be from unlabeled source AND below the biological CV threshold)
        train_mask = (gender_group['source'] == ccfg.unlabeled_label)
        
        # If your data has a 'cv' column, apply the project's strict filter:
        if 'igf_pnp_ratio_cv' in gender_group.columns:
            train_mask = train_mask & (gender_group['igf_pnp_ratio_cv'] < cv_limit) # select athletes beelow the CV threshold for the IGF-PNP ratio
        
        if train_mask.sum() > 0: 
            imp = SimpleImputer(strategy='median', keep_empty_features=True)
            rs = RobustScaler() 
            qt = QuantileTransformer(output_distribution='normal', n_quantiles=500, random_state=42)
            
            data_to_scale = gender_group[model_cols].values 
            imputed = imp.fit_transform(data_to_scale)
            
            # Fit ONLY on the filtered 'Super-Normal' reference athletes
            ref_data = imputed[train_mask.values]
            rs.fit(ref_data)
            qt.fit(rs.transform(ref_data))
            
            # Transform the entire gender group (Normal + Doped)
            transformed = qt.transform(rs.transform(imputed))
            gender_group.loc[:, model_cols] = transformed
            df_processed_list.append(gender_group)

    df_final = pd.concat(df_processed_list).sort_index()
    full_x = df_final[model_cols].values
    return df_final, full_x, model_cols

def standardize_scores(scores, invert=False): 
    if invert: scores = -scores 
    med = np.median(scores) 
    mad = np.median(np.abs(scores - med)) + 1e-6
    return (scores - med) / (mad * 1.4826)

if __name__ == "__main__":
    # --- 1. DATA PREPARATION ---
    df, full_x, model_cols = load_and_preprocess()
    
    print("[INFO] Generating Intra-individual Variance (ABP) features...")
    abp_df = abp.abp(df)
    df = df.merge(abp_df, on=['id', 'sex'], how="left")

    # --- NEW: AE ELBOW PLOT ---
    # This must be done BEFORE training to justify your choice of latent dimensions
    # It will save as '..._ae_elbow.png'
    viz.plot_ae_elbow(full_x, choice=6) 

    # --- 2. SSAE TRAINING (The "Nudge") ---
    print(f"\n--- Step 2: Training SSAE (Supervised Nudge) for {mcfg.epochs} Epochs ---")
    train_labels = (df['source'] == ccfg.positive_label).astype(int).values
    
    ae_scores, _, latent_full = ssae_mod.run_ssae(
        train_data=full_x, 
        full_data=full_x, 
        labels=train_labels, 
        epochs=mcfg.epochs
    )

    # Store nudged latent coordinates
    latent_df = pd.DataFrame(
        latent_full, 
        columns=[f'latent_dim_{i+1}' for i in range(latent_full.shape[1])],
        index=df.index
    )

    # --- 3. ENSEMBLE DETECTION ON NUDGED SPACE ---
    print("\n--- Step 3: Executing Ensemble Detectors on Latent Manifold ---")
    gh_mask = df['source'] == ccfg.positive_label
    doped_latent = latent_full[gh_mask]
    
    if_raw  = if_mod.run_iforest(latent_full, latent_full, ecfg)
    gmm_raw = gmm_mod.run_gmm(latent_full, latent_full, doped_latent, ecfg)
    svm_raw = svm_mod.run_svm(latent_full, train_labels, latent_full)

    # --- 4. SCORE AGGREGATION & STANDARDIZATION ---
    df['ae_score']  = standardize_scores(ae_scores)
    df['if_score']  = standardize_scores(if_raw, invert=True)
    df['gmm_score'] = standardize_scores(gmm_raw)
    df['svm_score'] = standardize_scores(svm_raw)
    
    w = ecfg.weights
    df['total_score'] = (
        df['ae_score']  * w['recon'] +
        df['if_score']  * w['iforest'] +
        df['gmm_score'] * w['gmm'] +
        df['svm_score'] * w['svm']
    )

    # --- 5. EXPORT RESULTS ---
    df_final = pd.concat([df, latent_df], axis=1)
    ssae_results_path = dcfg.final_results.replace(".csv", "_SSAE_Avenue.csv")
    df_final.to_csv(ssae_results_path, index=False)
    
    # Critical: Update config so viz pulls the NEW CSV with latent dims
    dcfg.final_results = ssae_results_path 

    # --- 6. FULL VISUALIZATION SUITE ---
    print("\n--- Step 4: Generating All Forensic Plots ---")
    
    #viz.plot_pca_elbow()  # Scree Plot (PCA of latent dims)
    #viz.plot_ae_elbow(full_x, choice=6)  #elbow plot of SSAE reconstruction error vs latent dim choice
    #viz.plot_performance()  # Wide 3-panel Performance (ROC/PR)
    #viz.plot_diagnostic_separation() # KDE "Nudge" verification
    viz.plot_tsne(latent_data = latent_full,
                  labels = train_labels,
                  df_metadata = df) # PR Curves for each model + ensemble   
    #viz.plot_model_comparison_pr()   
    #viz.plot_abp_map()


    print(f"\n[SUCCESS] Pipeline Complete. All plots generated in the results directory.")
