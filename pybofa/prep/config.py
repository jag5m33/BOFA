class data:
    merged_df = r'C:/Users/jagmeet/bofa_data/merged_df.csv'
    final_results = r'C:/Users/jagmeet/bofa_data/model_results/flagged_athletes.csv'
    latent_full = r'C:/Users/jagmeet/bofa_data/model_results/latent_full.npy'
class processor:
    # We use these for the initial cleanup and engineered ratio
    features = ['age', 'sex', 'avg_pnp', 'avg_igf', 'igf_pnp_ratio']
    pnp_cols = ['ln_pnp_orion', 'ln_pnp_cis', 'ln_pnp_siemens', 'ln_pnp_initial', 'ln_pnp_mean', 'avg_pnp']
    igf_cols = ['ln_igf_immuno', 'ln_igf_immulite', 'ln_igf_ms', 'ln_igf_ids', 'ln_igf_imt', 'ln_igf_initial', 'ln_igf_mean', 'avg_igf']

class model_params:
    epochs = 150
    latent_dim = 3    # Squeezing 5D input into 3D latent space
    patience = 25
    batch_size = 32

class ensemble_params:
    iforest_contam = 0.05 
    svm_nu = 0.05
    gmm_components = 5      
    random_state = 42
    
    weights = {
        'iforest': 0.40, # Good at catching isolated "spikes"
        'svm':     0.00, 
        'gmm':     0.40, # Good at catching clusters of weirdness
        'recon':   0.20  
    }

class calibration:
    target_recall = 0.70 
    positive_label = "GH_CONTROL"
    unlabeled_label = "ATHLETE_REF"