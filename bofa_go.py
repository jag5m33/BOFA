import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Imports from your new modular library
from pybofa.prep.config import data as dcfg, processor as pcfg, model_params as mcfg, ensemble_params as ecfg, calibration as ccfg
from pybofa.models.ae import build_and_train_ae, get_recon_error
from pybofa.models.IF import get_iforest_scores
from pybofa.models.SVM import get_svm_scores
from pybofa.models.gmm import get_gmm_scores

def load_data():
    df = pd.read_csv(dcfg.merged_df)
    # Maintain the engineered feature
    df['igf_pnp_ratio'] = df['avg_igf'] / (df['avg_pnp'] + 1e-6)
    
    features = pcfg.features # Loads from config: age, sex, pnp, igf, ratio
    df = df.dropna(subset=features)
    
    train_mask = df['source'] == 'ATHLETE_REF'
    scaler = StandardScaler().fit(df[train_mask][features])
    
    return df, scaler.transform(df[train_mask][features]), scaler.transform(df[features])

if __name__ == "__main__":
    # 1. Prepare Data
    df, train_scaled, full_scaled = load_data()

    # 2. Extract Latent Features (Autoencoder)
    ae_model, encoder = build_and_train_ae(
        train_scaled, mcfg.latent_dim, mcfg.epochs, mcfg.batch_size, mcfg.patience
    )
    latent_train = encoder.predict(train_scaled)
    latent_full = encoder.predict(full_scaled)
    recon_err = get_recon_error(ae_model, full_scaled)

    # 3. Generate Anomaly Scores
    if_s = get_iforest_scores(latent_train, latent_full, ecfg.iforest_contam, ecfg.random_state)
    svm_s = get_svm_scores(latent_train, latent_full, ecfg.svm_nu)
    gmm_s = get_gmm_scores(latent_train, latent_full, ecfg.gmm_components, ecfg.random_state)

    # 4. Standardized Weighted Voting
    def norm(s): return (s - s.mean()) / s.std()
    w = ecfg.weights
    
    df['total_suspicion'] = (
        norm(if_s) * w['iforest'] + 
        norm(svm_s) * w['svm'] + 
        norm(gmm_s) * w['gmm'] + 
        norm(recon_err) * w['recon']
    )

    # 5. Semi-Supervised Calibration Analysis
    gh_scores = df[df['source'] == 'GH_CONTROL']['total_suspicion']
    ath_scores = df[df['source'] == 'ATHLETE_REF']['total_suspicion']

    print(f"\n{'Target Recall':<15} | {'Athlete Suspicion Rate'}")
    print("-" * 40)
    for recall in [0.90, 0.80, 0.70, 0.60, 0.50]:
        thresh = np.percentile(gh_scores, (1 - recall) * 100)
        fpr = (ath_scores > thresh).mean()
        print(f"{recall*100:>12.0f}%      | {fpr*100:>18.1f}%")

    # 6. Final Labeling
    final_thresh = np.percentile(gh_scores, (1 - ccfg.target_recall) * 100)
    df['final_flag'] = df['total_suspicion'] > final_thresh
    
    # Save the results
    df.to_csv(dcfg.final_results, index=False)
    print(f"\nProcess Complete. Results saved to: {dcfg.final_results}")