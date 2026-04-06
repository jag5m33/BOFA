class data:
    merged_df = r'C:/Users/jagmeet/bofa_data/merged_df.csv'
    final_results = r'C:/Users/jagmeet/bofa_data/model_results/flagged_athletes.csv'
    latent_full = r'C:/Users/jagmeet/bofa_data/model_results/latent_full.csv'
class processor:
    # We use these for the initial cleanup and engineered ratio
    features = ['age', 'sex', 'avg_pnp', 'avg_igf', 'igf_pnp_ratio']
    pnp_cols = ['ln_pnp_orion', 'ln_pnp_cis', 'ln_pnp_siemens', 'ln_pnp_initial', 'ln_pnp_mean', 'avg_pnp']
    igf_cols = ['ln_igf_immuno', 'ln_igf_immulite', 'ln_igf_ms', 'ln_igf_ids', 'ln_igf_imt', 'ln_igf_initial', 'ln_igf_mean', 'avg_igf']

class model_params:
    epochs = 150
    latent_dim = 6    # Squeezing 5D input into 3D latent space
    patience = 25
    batch_size = 32
    l1_reg = 1e-5

class ensemble_params:
    gmm_components = 4
    random_state = 42
    iforest_contam = 0.05
    svm_nu = 0.05
    
    # Adjusted based on your performance table
    weights = {
        'svm': 0.50,      # Increased: Since this has your best PR stats, it should lead.
        'gmm': 0.35,      # Decreased: Use this to provide the "biological tail" logic.
        'iforest': 0.10,  # Minimal: Keep as a tiny safety net for extreme outliers only.
        'recon': 0.05      # Keep at 0.00 for now to see if the latent space models stabilize.
    }
class calibration:
    target_recall = 0.70 
    positive_label = "GH_CONTROL"
    unlabeled_label = "ATHLETE_REF"

class shades:
    C_BLUE = "#0072B2"    # Male / Reference
    C_ORANGE = "#D55E00"  # Female / Contrast
    C_GREEN = "#009E73"   # Success / Ensemble
    C_BLACK = "#000000"   # Doped / Outlier
