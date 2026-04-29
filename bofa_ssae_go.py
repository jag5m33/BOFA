import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import shap

# --- 1. CONFIG & MODEL IMPORTS ---
from pybofa.prep.config import (
    data as dcfg, processor as pcfg, 
    features as fcfg, biology as bcfg
)
from pybofa.plots import bofa_viz as viz    
import pybofa.models.ssae as ssae_mod  
import pybofa.models.SVM as svm_mod
import pybofa.models.ls as ls_mod 
import pybofa.models.abp as abp

def load_and_preprocess():
    """
    Step 1: Biological Cleaning Pipeline.
    Uses Median Imputation and Normalization to prep for the SSAE.
    """
    print("--- Step 1: Loading & Cleaning Biological Markers ---")
    df = pd.read_csv(dcfg.merged_df)
    
    # Standardize time and sort for longitudinal pathing
    df[fcfg.date_col] = pd.to_datetime(df[fcfg.date_col])
    df = df.sort_values([fcfg.id_col, fcfg.date_col])
    
    # feature engineering: IGF-1 to PNP ratio (a known GH doping signature)
    df['igf_pnp_ratio'] = df['avg_igf'] / df['avg_pnp']
    
    # 1. athlete biological passport metrics ABP - calculated and merged as additional features to df dataset
    abp_metrics = abp.abp(df) 
    df = df.merge(abp_metrics, on=['id', 'sex'], how="left")
    
    # 2. Define Training Features
        # add lists together with dict.fromkeys() creates a dictionary with UNIQUE keys 
        # this prevents duplicates while preserving order, so 'igf_pnp_ratio' is only added once
        #list(dict.fromkeys()) is a common Python trick to remove duplicates while preserving order in a list.
    bio_cols = list(dict.fromkeys(pcfg.pnp_cols + pcfg.igf_cols + ['igf_pnp_ratio']))
    # dont use sets for this, you loose the order of the variables
    
    #  identify the new ABP columns here:
    vol_cols = [c for c in df.columns if 'volatility' in c]    
    # establish final feature list for models - use if the ssae should see volaitlity orj ust snapshots
    feature_list = bio_cols + ['sex'] + vol_cols

    # from the origonal dataset remove nan and replace with 0
    df[vol_cols] = df[vol_cols].fillna(0)
    
    # 3. median imputation: removes NaN values wihtout skewing dsitribution (critical for small forensic datasets)
    imputer = SimpleImputer(strategy='median')
    df[feature_list] = imputer.fit_transform(df[feature_list])

    
    # 4. z-score normalisation: Scales markers to Z-distribution (Mean=0, Std=1)
    # prevents IGF-1 from dominating the 3D Bottleneck
    scaler = StandardScaler()
    full_x = scaler.fit_transform(df[feature_list]) 
    
    return df, full_x, feature_list

# df is the full dataset 
# full_x is the cleaned and preprocessed feature matrix ready for modeling
# feature_list is the list of features used in the model, which can be useful for later reference in visualizations and analysis. also including vol_cols 

def robust_standardize(scores, labels):
    # Determine 'Normal' bounds using only Reference athletes
        # for all 0 labels (reference athletes), calculate the median and IQR to define a robust scale
    baseline = scores[labels == 0]
    med = np.median(baseline)
    q75, q25 = np.percentile(baseline, [75, 25])
    iqr = q75 - q25 + 1e-7
    
    # apply scale to all
    return (scores - med) / iqr

def normalize_for_viz(series):
    """
    Log-then-Standardize sequence specifically for KDE plots.
    This stretches the 'needle peak' and exposes the doping hump.
    """
    shifted = series - series.min() + 1.0
    logged = np.log1p(shifted)
    return (logged - logged.mean()) / logged.std()

if __name__ == "__main__":

    # 1. call proc and model defs.
    df, full_x, feature_names = load_and_preprocess()
    
    # Define Ground Truth (GH_ADMIN and POSITIVE are Class 1)
        #label = FIRST PART IS GH_CONTROL (1) and ATHLETE_REF (0) is the reference class. 
        # This binary label is used for training the models, where GH_CONTROL samples are treated as 'positive' cases of doping, and ATHLETE_REF samples are treated as 'negative' or 'unlabeled' cases.
    labels = df['source'].isin(['GH_CONTROL']).astype(int).values

    # 2. calling and running defs for models
    print("--- Step 2: Running Original Models ---")
    
    # Model A: SSAE (saving reconstructed_x MSE for Heatmap)
    ae_raw, latent_full, model, recon_x, encoder, history, shap_values, x_test  , background_array= ssae_mod.run_ssae(full_x, full_x, labels)
    
    # Model B: Weighted one-class SVM (Genomic hyperplane separation in the 3D Manifold)
    svm_raw = svm_mod.run_svm(latent_full, labels, latent_full)
    
    # Model C: Label Spreading (Graph-based suspicious athlete propagation on the 3D Manifold)
    ls_raw = ls_mod.run_label_spreading(latent_full, labels)

    # 3. integrate scores together and evaluate 
    print("--- Step 3: Creating Consensus ---")
        # calculate normalised version of scores with robust_standardise using normal reference population
        # This ensures that the final scores are on a comparable scale and that the ensemble average is meaningful.
    df['ae_score'] = robust_standardize(ae_raw, labels)
    df['svm_score'] = robust_standardize(svm_raw, labels)
    df['ls_score'] = ls_raw
    
    # Consensus Average (Overall athlete suspicion score): simple average of the three model scores, then re-standardized for interpretability
    df['total_score'] = (df['ae_score'] + df['svm_score'] + df['ls_score']) / 3

    df['total_score'] = robust_standardize(df['total_score'], labels)

    # KDE scores (Log-then-Standardise)
    df['ae_viz'] = normalize_for_viz(df['ae_score'])
    df['svm_viz'] = normalize_for_viz(df['svm_score'])
    df['ls_viz'] = normalize_for_viz(df['ls_score'])
    df['total_viz'] = normalize_for_viz(df['total_score'])

    # remove any additional NANs that have been ignores or missed in the current list generated
    score_cols = ['ae_score', 'svm_score', 'ls_score', 'total_score', 'total_viz']
        # np.inf, or -np.inf remove infinity values and replaces with 0 in any of the score cols in the list above.
    df[score_cols] = df[score_cols].fillna(0).replace([np.inf, -np.inf], 0)

    # 4. run visualisation script 
    print("--- Step 4: generating dissertation graphics ---")
    viz.generate_all_plots(
        df=df, 
        latent_full=latent_full, 
        full_x=full_x, 
        reconstructed_x=recon_x, 
        labels=labels, 
        feature_names=feature_names,
        encoder=encoder,
        history=history, 
        scores=ae_raw,
        shap_values = shap_values,
        x_test = x_test,
        background_array = background_array, 
        model = model    
    )
    # 5. SAVE RESULTS
    df.to_csv(dcfg.final_results, index=False)
    print(f"\n[SUCCESS] Pipeline complete. Results saved: {dcfg.final_results}")