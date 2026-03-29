import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Imports from your new modular library
from pybofa.prep.config import data as dcfg, processor as pcfg, model_params as mcfg, ensemble_params as ecfg, calibration as ccfg
from pybofa.models.ae import build_and_train_ae, get_recon_error
from pybofa.models.IF import get_iforest_scores
from pybofa.models.SVM import get_svm_scores
from pybofa.models.gmm import get_gmm_scores

# RUN MODELS ON ATHETE_REF DATA AND PREDICT ON EVERYTHING.

# cloads and cleans the data 
def load_data():
    df = pd.read_csv(dcfg.merged_df)
    # Maintain the engineered feature
    df['igf_pnp_ratio'] = df['avg_igf'] / (df['avg_pnp'] + 1e-6) # add an additonal column to capture the ratio of IGF-I to P-III-NP, which is a key biological signal in doping detection. 
    # This ratio can help capture the 'tilt' in an athlete's physiology that doping might cause.
    
    features = pcfg.features # Loads from config: age, sex, pnp, igf, ratio
    df = df.dropna(subset=features) # drop any rows with missing values in these features to ensure clean training data
    
    train_mask = df['source'] == 'ATHLETE_REF' # create a DF: using source column; isolate control + unknown
    scaler = StandardScaler().fit(df[train_mask][features]) # scale the feature columns from athlete_ref from df
    
    return df, scaler.transform(df[train_mask][features]), scaler.transform(df[features])

if __name__ == "__main__": # only run script when executed directly (not imported as a module)
    # 1. Prepare Data
    df, train_scaled, full_scaled = load_data() # load definition, create the df used in the ae and ml models

    # 2. Extract Latent Features (Autoencoder)
    ae_model, encoder = build_and_train_ae(
        train_scaled, mcfg.latent_dim, mcfg.epochs, mcfg.batch_size, mcfg.patience
    )
    latent_train = encoder.predict(train_scaled)
    latent_full = encoder.predict(full_scaled)
    recon_err = get_recon_error(ae_model, full_scaled) # Measures how well the ae reconstructs each sample
        # ae model - engage the full model, on all the scaled data (athlete + contol) 
        # to get the recon error for each sample (higher error = more likely to be doped)

    #Run Each Model#

    # 3. Generate Anomaly Scores
    if_s = get_iforest_scores(latent_train, latent_full, ecfg.iforest_contam, ecfg.random_state)
    svm_s = get_svm_scores(latent_train, latent_full, ecfg.svm_nu)
    gmm_s = get_gmm_scores(latent_train, latent_full, ecfg.gmm_components, ecfg.random_state)

    # 4. Standardized Weighted Voting
    def norm(s): return (s - s.mean()) / s.std() # standardise Z scores (individually per column)

    w = ecfg.weights
    
    # Weighting: call the normalisation function and pass in scores of each model + multiply with weight from config
    df['total_suspicion'] = (
        norm(if_s) * w['iforest'] + 
        norm(svm_s) * w['svm'] + 
        norm(gmm_s) * w['gmm'] + 
        norm(recon_err) * w['recon']
    )

    # 5. Semi-Supervised Calibration Analysis
    gh_scores = df[df['source'] == 'GH_CONTROL']['total_suspicion'] # object with all GH_control athletes identified with suspician score
    ath_scores = df[df['source'] == 'ATHLETE_REF']['total_suspicion'] # object with all ATHLETE_REF athletes identified with suspician score

    print(f"\n{'Target Recall':<15} | {'Athlete Suspicion Rate'}") # calibration table: shows the tradeoff between catching more doped athletes (recall) vs accidentally flagging clean athletes 
        # (suspicion rate)
    print("-" * 40)
    for recall in [0.90, 0.80, 0.70, 0.60, 0.50]: # testing different recall sensitivity levels (how many doped athletes we catch) 
            #and see the corresponding suspicion rate in clean athletes

        thresh = np.percentile(gh_scores, (1 - recall) * 100) # if recall= 90% --> keep top 90% of GH cases
        fpr = (ath_scores > thresh).mean()  # athletes flagged as suspicious above this threshold / total number of clean athletes (false positive rate)
        print(f"{recall*100:>12.0f}%      | {fpr*100:>18.1f}%")

    # 6. Final Labeling
    final_thresh = np.percentile(gh_scores, (1 - ccfg.target_recall) * 100) # sets decision boundary
    df['final_flag'] = df['total_suspicion'] > final_thresh
    
    # Save the results
    df.to_csv(dcfg.final_results, index=False)
    print(f"\nProcess Complete. Results saved to: {dcfg.final_results}")