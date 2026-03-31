import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, auc

# Custom Configs (Keep as they are in your system)
from pybofa.prep.config import data as dcfg
from pybofa.prep.config import model_params as mcfg
from pybofa.prep.config import ensemble_params as ecfg
from pybofa.prep.config import processor as pcfg

# Custom Models (ae.py needs the update above)
import pybofa.models.ae as ae_mod
import pybofa.models.IF as if_mod    
import pybofa.models.gmm as gmm_mod 
import pybofa.models.SVM as svm_mod

def load_and_preprocess():
    print("Loading and Preprocessing Full-Spectrum Assay Data...")
    df = pd.read_csv(dcfg.merged_df)
    
    # 1. Feature Engineering
    df['igf_pnp_ratio'] = df['avg_igf'] - df['avg_pnp']
    
    # 2. Define Features
    model_cols = pcfg.pnp_cols + pcfg.igf_cols + ['igf_pnp_ratio']
    
    df_processed_list = []

    # 3. Strict Gender-Specific Normalization
    for s in df['sex'].unique():
        gender_group = df[df['sex'] == s].copy()
        train_mask = (gender_group['source'] == 'ATHLETE_REF')
        
        if train_mask.sum() > 0:
            imp = SimpleImputer(strategy='median', keep_empty_features=True)
            rs = RobustScaler()
            qt = QuantileTransformer(output_distribution='normal', n_quantiles=min(len(gender_group), 100))
            
            data = gender_group[model_cols].values
            imputed = np.nan_to_num(imp.fit_transform(data), nan=0.0)
            transformed = qt.fit_transform(rs.fit_transform(imputed))
            
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
    
    # 2. Build High-Dimensional Latent Space (RUN ONCE)
    print(f"Building {mcfg.latent_dim}D Latent Space...")
    ae_scores, latent_train, latent_full = ae_mod.run_ae(train_x, full_x, mcfg)

    # =========================================================================
    # NEW: Call the 3D Visualization with Gender Separation & Highlights
    # =========================================================================
    # This colors points by Source, splits by Gender (shape/alpha), 
    # and highlights Male GH_CONTROL
    ae_mod.visualize_latent_3d_gender(latent_full, df, save_path="latent_3d_gender_map.png")
    # =========================================================================
    
    # 3. Feed Latent Space to the Ensemble
    print(f"Running Ensemble on Latent Map...")
    if_raw  = if_mod.run_iforest(latent_train, latent_full, ecfg)
    gmm_raw = gmm_mod.run_gmm(latent_train, latent_full, ecfg)
    svm_raw = svm_mod.run_svm(latent_train, latent_full, ecfg)
    
    # 4. Standardize & Aggregate Scores
    df['ae_score']  = standardize_scores(ae_scores, invert=False)
    df['if_score']  = standardize_scores(if_raw,    invert=True)
    df['gmm_score'] = standardize_scores(gmm_raw,   invert=True)
    df['svm_score'] = standardize_scores(svm_raw,   invert=True)
    
    df['total_score'] = (
        df['ae_score']  * ecfg.weights.get('recon', 0.20) +
        df['if_score']  * ecfg.weights.get('iforest', 0.40) +
        df['gmm_score'] * ecfg.weights.get('gmm', 0.40)
    )

    # 5. Consensus Flags (Required for the Recall table below)
    models = ['ae', 'if', 'gmm', 'svm']
    for m in models:
        thresh = np.percentile(df[f'{m}_score'], 95)
        df[f'{m}_flag'] = (df[f'{m}_score'] > thresh).astype(int)
    df['consensus_score'] = df[[f'{m}_flag' for m in models]].sum(axis=1)

    # =========================================================================
    # 6. RESULTS ANALYSIS (The Recall Table)
    # =========================================================================
    gh_mask = df['source'] == "GH_CONTROL"
    ath_mask = df['source'] == "ATHLETE_REF"

    # --- TABLE 1: Discrete Consensus Recall ---
    print("\n" + "="*75)
    print(f"{'Consensus Agreement Level':<25} | {'Recall (Doped)':<15} | {'Ath Suspicion'}")
    print("-" * 75)
    for level in range(1, 5):
        recall = (df[gh_mask]['consensus_score'] >= level).mean()
        suspicion = (df[ath_mask]['consensus_score'] >= level).mean()
        print(f"Flagged by {level}+ Model(s) | {recall:>14.1%} | {suspicion:>17.1%}")
    print("="*75)

    # --- Overall System AUC ---
    y_true = (df['source'] == "GH_CONTROL").astype(int)
    fpr, tpr, _ = roc_curve(y_true, df['total_score'])
    print(f"\nOverall System AUC: {auc(fpr, tpr):.4f}")

    # 7. Final Exports
    print("Saving final results...")
    df.to_csv(dcfg.final_results, index=False)
    np.save(dcfg.latent_full, latent_full)