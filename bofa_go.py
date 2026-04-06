import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.impute import SimpleImputer

# Precise imports based on your file structure
from pybofa.prep.config import data as dcfg
from pybofa.prep.config import model_params as mcfg
from pybofa.prep.config import ensemble_params as ecfg
from pybofa.prep.config import processor as pcfg
from pybofa.prep.config import calibration as ccfg

# Model Imports
import pybofa.models.ae as ae_mod
import pybofa.models.IF as if_mod    
import pybofa.models.gmm as gmm_mod 
import pybofa.models.SVM as svm_mod
import pybofa.models.abp as abp
from pybofa.plots import bofa_viz as viz



def load_and_preprocess():
    """
    Loads raw data and performs gender-specific normalization 
    based on the Reference Population (ATHLETE_REF).
    """
    print(f"--- Step 1: Loading Data from {dcfg.merged_df} ---")
    df = pd.read_csv(dcfg.merged_df) # merged_df = df
    
    # 1. Feature Engineering: Log-ratio of IGF-1 to P-III-NP
    # subtraction in a log space is the same as ratio in a linear 
    df['igf_pnp_ratio'] = df['avg_igf'] - df['avg_pnp'] # create a new list in the df with average log difference of pnp-igf
    
    # 2. Build Feature List from Config
    model_cols = pcfg.pnp_cols + pcfg.igf_cols + ['igf_pnp_ratio']
    
    df_processed_list = []

    # 3. Gender-Specific Normalization Loop
    for s in df['sex'].unique():
        gender_group = df[df['sex'] == s].copy() # create a copy of the df with for each sex
        # Identify the 'Clean' reference athletes for scaling
        train_mask = (gender_group['source'] == ccfg.unlabeled_label) # pull out all gender groups with source of athlete_ref
        
        if train_mask.sum() > 0: 
            # Impute missing lab results using median values so model doesnt crash on empty cells
            imp = SimpleImputer(strategy='median', keep_empty_features=True)
            
            # Scale with regards to gender and robustly (using IQR instead of mean/std)
            # median values are often blind to outliers
            rs = RobustScaler()             

            # forces data into perfect bell curve (normal distribution) ideal for GMM and AE
            qt = QuantileTransformer(
                output_distribution='normal', 
                n_quantiles=min(len(gender_group), 500), # min of 500 or samples to avoid errors
                random_state=42 # Fixed: added hardcoded seed to ensure reproducibility
            )
            
            # data is the subset of the df with only the columns we want to use for modeling
            data = gender_group[model_cols].values 
            
            # Impute, Scale, then Map to Normal Distribution
            imputed = np.nan_to_num(imp.fit_transform(data), nan=0.0) # replace NAN with medians 
            
            # Quantile transformer applied to imputed dataset & robust scaler
            transformed = qt.fit_transform(rs.fit_transform(imputed))
            
            # Overwrites raw lab values with new bell curve mapped values for outlier detection
            gender_group[model_cols] = transformed
            
            # Processed male and female separately - save to list
            df_processed_list.append(gender_group)

    # Stacks dataframes back together into one final df preserving original row order
    df_final = pd.concat(df_processed_list).sort_index()
    
    # Split into Training (Ref) and Full (All athletes)
    train_x = df_final[df_final['source'] == ccfg.unlabeled_label][model_cols].values
    full_x = df_final[model_cols].values
    
    return df_final, train_x, full_x


def standardize_scores(scores, invert=False): 
    """
    Takes raw output from different models and converts to Z-score using 
    Median Absolute Deviation (MAD) for robustness.
    """
    if invert: scores = -scores # Higher value must always = more anomalous 
    
    med = np.median(scores) 
    # Use 1.4826 as the scaling factor to make MAD consistent with standard deviation
    mad = np.median(np.abs(scores - med)) + 1e-6
    
    return (scores - med) / (mad * 1.4826)


if __name__ == "__main__": 
    # 1. Load and Preprocess
    df, train_x, full_x = load_and_preprocess()
    
    print("STEP 1.5: generating intra-individual variance (ABP) features...")
    abp_df = abp.abp(df)
    # new format of the merged dataset improved, shows which athletes are flagged for intra-individual variance
    df = df.merge(abp_df, on=['id', 'sex'], how = "left")

    # 2. Run Autoencoder
    print(f"--- Step 2: Training Autoencoder ---")
    ae_scores, latent_train, latent_full = ae_mod.run_ae(train_x, full_x, mcfg.epochs)

    # 3. SAVE TO CSV
    latent_df = pd.DataFrame(
        latent_full, 
        columns=[f'latent_dim_{i+1}' for i in range(latent_full.shape[1])]
    )
    latent_df['sex'] = df['sex'].values
    latent_df['source'] = df['source'].values
    
    csv_out = dcfg.latent_full.replace('.npy', '.csv') 
    print(f"Exporting latent space to: {csv_out}")
    latent_df.to_csv(csv_out, index=False)
    
    
    # --- 4. SEMI-SUPERVISED ENSEMBLE DETECTION (OPERATING ON LATENT SPACE) ---
    print("\n--- Step 3: Executing Semi-Supervised Ensemble Detectors ---")
    
    # Extract known Doped samples in the latent space to guide GMM/SVM
    gh_mask = df['source'] == ccfg.positive_label
    doped_latent = latent_full[gh_mask]
    
    # Create labels for SVC (1 = Doped, 0 = Reference)
    y_semi = (df['source'] == ccfg.positive_label).astype(int).values
    
    # Model Execution
    if_raw  = if_mod.run_iforest(latent_train, latent_full, ecfg)
    gmm_raw = gmm_mod.run_gmm(latent_train, latent_full, doped_latent, ecfg)
    svm_raw = svm_mod.run_svm(latent_full, y_semi, latent_full)
    
    # --- 5. SCORE AGGREGATION ---
    # Isolation Forest: More negative = More outlier. INVERT = TRUE
    df['if_score']  = standardize_scores(if_raw, invert=True)

    # GMM: Now using Contrastive Likelihood Ratio (Higher = More Doped). INVERT = FALSE
    df['gmm_score'] = standardize_scores(gmm_raw, invert=False)
    # Autoencoder: Higher Reconstruction Error = More outlier. INVERT = FALSE
    df['ae_score']  = standardize_scores(ae_scores, invert=False)

    # SVM/SVC: Higher probability of Class 1 = More Doped. INVERT = FALSE
    df['svm_score'] = standardize_scores(svm_raw, invert=False)
    
    # Apply weights strictly from config
    w = ecfg.weights
    df['total_score'] = (
        df['ae_score']  * w['recon'] +
        df['if_score']  * w['iforest'] +
        df['gmm_score'] * w['gmm'] +
        df['svm_score'] * w['svm']
    )

    # --- 6. CONSENSUS & RECALL ANALYSIS ---
    # Flag top 5% anomalies for each model
    models = ['ae', 'if', 'gmm', 'svm']
    for m in models:
        # Calculate 95th percentile threshold
        thresh = np.percentile(df[f'{m}_score'], 95)
        # Create flag column
        df[f'{m}_flag'] = (df[f'{m}_score'] > thresh).astype(int)
    
    # Consensus Score: How many models flagged the athlete (0 to 4)
    df['consensus_score'] = df[[f'{m}_flag' for m in models]].sum(axis=1)

    # Performance Masks
    gh_mask = df['source'] == ccfg.positive_label
    ath_mask = df['source'] == ccfg.unlabeled_label

    print("\n" + "="*75)
    print(f" Dissertation performance: Recall TABLE (Target: {ccfg.target_recall:.0%})")
    print("-" * 75)
    print(f"{'Agreement Level':<25} | {'Recall (Doped)':<15} | {'Athlete Suspicion'}")
    print("-" * 75)
    
    for level in range(1, 5):
        # Recall: Proportion of GH_CONTROL correctly flagged at this level
        recall = (df[gh_mask]['consensus_score'] >= level).mean()
        # Suspicion: Proportion of ATHLETE_REF flagged (False Positive Rate)
        suspicion = (df[ath_mask]['consensus_score'] >= level).mean()
        print(f"Flagged by {level}+ Model(s) | {recall:>14.1%} | {suspicion:>17.1%}")
    print("="*75)

    # --- 7. FINAL EXPORT ---
    print(f"Saving final suspected athlete list to: {dcfg.final_results}")
    df.to_csv(dcfg.final_results, index=False)
    
    print("\n[SUCCESS] Pipeline complete. Check plots for final analysis.")
    
        #save latent_dim for the visualization step
    # ---  FINAL EXPORT ---
    # 1. Create a DataFrame for the coordinates found by the Autoencoder
    latent_df = pd.DataFrame(
        latent_full, 
        columns=[f'latent_dim_{i+1}' for i in range(latent_full.shape[1])],
        index=df.index
    )

    # Combine everything into one dataframe BEFORE saving
    df_final = pd.concat([df, latent_df], axis=1)

    # SAVE TO DISK NOW so the viz functions can see the new columns
    print(f"Exporting full results to {dcfg.final_results}")
    df_final.to_csv(dcfg.final_results, index=False)

    # --- 5. VISUALIZATION (Must happen AFTER .to_csv) ---
    print("\n[INFO] Starting Visualization Pipeline...")
    viz.plot_method_logic() 
    #viz.plot_elbow_justification()
    #viz.plot_ae_elbow(train_x, mcfg.latent_dim)
    viz.plot_3d_tsne()      
    viz.plot_results()     
    #viz.plot_cm(target_recall=ccfg.target_recall) 
    viz.plot_abp_suspicion_map()

    print("[SUCCESS] All plots updated.")