import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
from pybofa.prep.config import data as dcfg
from pybofa.prep.config import model_params as mcfg
from pybofa.prep.config import ensemble_params as ecfg
from pybofa.prep.config import processor as pcfg

import pybofa.models.ae as ae_mod
import pybofa.models.IF as if_mod    
import pybofa.models.gmm as gmm_mod 
import pybofa.models.SVM as svm_mod

def load_and_preprocess():
    print("Loading and Preprocessing Full-Spectrum Assay Data...")
    df = pd.read_csv(dcfg.merged_df)
    
    # 1. Feature Engineering
    df['igf_pnp_ratio'] = df['avg_igf'] - df['avg_pnp']
    
    # 2. Define the Full Biological Marker List
    pnp_cols = pcfg.pnp_cols
    igf_cols = pcfg.igf_cols
    
    model_cols = pnp_cols + igf_cols + ['igf_pnp_ratio']
    
    df_processed_list = []

    # 3. Strict Gender-Specific Normalization
    for s in df['sex'].unique():
        gender_group = df[df['sex'] == s].copy()
        train_mask = (gender_group['source'] == 'ATHLETE_REF')
        
        if train_mask.sum() > 0:
            # Use constant fill for totally empty columns to prevent dropping features
            # 'keep_empty_features=True' ensures we always return 15 columns
            imp = SimpleImputer(strategy='median', keep_empty_features=True)
            rs = RobustScaler()
            qt = QuantileTransformer(output_distribution='normal', n_quantiles=min(len(gender_group), 100))
            
            # Extract data
            data_to_process = gender_group[model_cols].values
            
            # Step-by-step transformation
            # We use .values to avoid index/column alignment issues during transformation
            imputed = imp.fit_transform(data_to_process)
            
            # Handle any remaining NaNs (if a column was entirely NaN, median imputer might leave it as is)
            if np.isnan(imputed).any():
                imputed = np.nan_to_num(imputed, nan=0.0)
                
            scaled = rs.fit_transform(imputed)
            transformed = qt.fit_transform(scaled)
            
            # Re-assign back to the dataframe
            gender_group[model_cols] = transformed
            df_processed_list.append(gender_group)
    
    df_final = pd.concat(df_processed_list).sort_index()
    train_x = df_final[df_final['source'] == 'ATHLETE_REF'][model_cols].values
    full_x = df_final[model_cols].values
    
    return df_final, train_x, full_x

def standardize_scores(scores, invert=False):
    if invert: scores = -scores
    med = np.median(scores)
    mad = np.median(np.abs(scores - med)) + 0.01
    return np.clip((scores - med) / mad, -10, 10)

if __name__ == "__main__":
    # 1. Prepare Data
    df, train_x, full_x = load_and_preprocess()
    
    # 2. Build High-Dimensional Latent Space
    # Now mapping ~15 features into the Bottleneck
    print(f"Building {mcfg.latent_dim}D Latent Space from {full_x.shape[1]} biological markers...")
    ae_scores, latent_train, latent_full = ae_mod.run_ae(train_x, full_x, mcfg)
    
    # 3. STEP TWO: Feed Latent Space to the Ensemble
    print(f"Running Ensemble on High-Dimensional Latent Map...")
    if_raw  = if_mod.run_iforest(latent_train, latent_full, ecfg)
    gmm_raw = gmm_mod.run_gmm(latent_train, latent_full, ecfg)
    svm_raw = svm_mod.run_svm(latent_train, latent_full, ecfg)
    
    # 4. Standardize & Align
    # ... [previous code for scaling and individual model runs] ...

    # 4. Standardize & Align
    df['ae_score']  = standardize_scores(ae_scores, invert=False)
    df['if_score']  = standardize_scores(if_raw,    invert=True)
    df['gmm_score'] = standardize_scores(gmm_raw,   invert=True)
    df['svm_score'] = standardize_scores(svm_raw,   invert=True)
    
    # NEW: Calculate the Weighted "Total Score"
    # This uses the weights from your ensemble_params config
    df['total_score'] = (
        df['ae_score']  * ecfg.weights.get('recon', 0.20) +
        df['if_score']  * ecfg.weights.get('iforest', 0.40) +
        df['gmm_score'] * ecfg.weights.get('gmm', 0.40) +
        df['svm_score'] * ecfg.weights.get('svm', 0.00)
    )

    # 5. Strict Consensus Flags (Top 5% per model)
    models = ['ae', 'if', 'gmm', 'svm']
    for m in models:
        thresh = np.percentile(df[f'{m}_score'], 95)
        df[f'{m}_flag'] = (df[f'{m}_score'] > thresh).astype(int)
    df['consensus_score'] = df[[f'{m}_flag' for m in models]].sum(axis=1)

    # 6. RESULTS ANALYSIS
    gh_mask = df['source'] == "GH_CONTROL"
    ath_mask = df['source'] == "ATHLETE_REF"

    # --- TABLE 1: Discrete Consensus (What you saw before) ---
    print("\n" + "="*75)
    print(f"{'Consensus Agreement Level':<25} | {'Recall (Doped)':<15} | {'Ath Suspicion'}")
    print("-" * 75)
    for level in range(1, 5):
        recall = (df[gh_mask]['consensus_score'] >= level).mean()
        suspicion = (df[ath_mask]['consensus_score'] >= level).mean()
        print(f"Flagged by {level}+ Model(s) | {recall:>14.1%} | {suspicion:>17.1%}")
    print("="*75)

    # --- TABLE 2: Total Weighted Ensemble Performance ---
    # We find the threshold on the 'total_score' that catches 70% of doped samples
    target_rec = 0.70  # Or use calibration.target_recall
    doped_scores = df[gh_mask]['total_score']
    # Threshold is the value where (1 - percentile) = target_recall
    total_thresh = np.percentile(doped_scores, (1 - target_rec) * 100)
    
    total_recall = (df[gh_mask]['total_score'] >= total_thresh).mean()
    total_suspicion = (df[ath_mask]['total_score'] >= total_thresh).mean()

    print("\n" + "="*75)
    print(f"{'SYSTEM TOTAL (Weighted)':<25} | {'Recall (Doped)':<15} | {'Ath Suspicion'}")
    print("-" * 75)
    print(f"Full Ensemble Score       | {total_recall:>14.1%} | {total_suspicion:>17.1%}")
    print(f"Threshold used: {total_thresh:.2f}")
    print("="*75)

    df.to_csv(dcfg.final_results, index=False)