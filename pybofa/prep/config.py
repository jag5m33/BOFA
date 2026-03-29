# ================================================================
# CONFIGURATION FOR SEMI-SUPERVISED ANTI-DOPING ML
# ================================================================

class data:

    merged_df = r'C:/Users/jagmeet/bofa_data/merged_df.csv'
    # Result storage
    latent_space = r'C:/Users/jagmeet/bofa_data/model_results/athlete_latent_space.csv'
    final_results = r'C:/Users/jagmeet/bofa_data/model_results/flagged_athletes.csv'


class processor:

    # Core GH-2000 features + engineered ratio to capture physiological 'tilt'
    features = ['age', 'sex', 'avg_pnp', 'avg_igf', 'igf_pnp_ratio']
    validation_split = 0.2


class model_params:

    epochs = 100
    latent_dim = 3  # Compressed representation for anomaly models
    patience = 10
    batch_size = 32


class ensemble_params:

    # We use very low contamination to focus only on extreme outliers
    iforest_contam = 0.005 
    svm_nu = 0.05
    gmm_components = 4
    random_state = 42

    # MODEL WEIGHTING (The total should equal 1.0)
    # GMM and IForest often capture the IGF-I/P-III-NP relationship best
    weights = {
        'iforest': 0.30,
        'svm':     0.15,
        'gmm':     0.35,
        'recon':   0.20  
    }

class calibration:
  
    # Percentile of GH_CONTROL we target catching (Recall)
    # Lower this (e.g., to 0.50) if the Athlete Suspicion Rate is too high
    target_recall = 0.60 
    
    # Maximum acceptable suspicion rate in the clean athlete population
    max_athlete_fpr = 0.05