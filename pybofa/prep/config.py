import os

# --- 1. DATA PATHS ---
class data:
    merged_df = r'C:/Users/jagmeet/bofa_data/merged_df.csv'
    final_results = r'C:/Users/jagmeet/bofa_data/model_results/flagged_athletes.csv'
    latent_full = r'C:/Users/jagmeet/bofa_data/model_results/latent_full.csv'

# --- 2. BIOMARKER & COLUMN DEFINITIONS ---
class processor:
    # Added igf_pnp_ratio here so the Go script picks it up automatically
    features = ['age', 'sex', 'avg_pnp', 'avg_igf', 'igf_pnp_ratio']
    pnp_cols = ['ln_pnp_orion', 'ln_pnp_cis', 'ln_pnp_siemens', 'ln_pnp_initial', 'ln_pnp_mean', 'avg_pnp']
    igf_cols = ['ln_igf_immuno', 'ln_igf_ms', 'ln_igf_ids', 'ln_igf_imt', 'ln_igf_initial', 'ln_igf_mean']
    igf_pnp_cols = ['igf_pnp_ratio']

# --- 3. CORE MODEL ARCHITECTURE ---
class model_params:
    epochs = 150
    latent_dim = 6
    patience = 25
    batch_size = 32
    random_state = 42
    l1_reg = 1e-4   # Matches the SSAE script's sparsity requirement

# --- 4. FORENSIC WEIGHTING ---
class ssae:
    reconstruction = 1.0 
    classifier_weight = 0.0001  

class ensemble_params:
    w_svm = 0.40 
    w_ae  = 0.40 
    w_ls  = 0.20 
    
    svm_C = 500.0     
    svm_gamma = 0.01  
    svm_weight = 50.0 
    
    ls_neighbors = 3        
    ls_alpha = 0.1    

# --- 5. FORENSIC BIOLOGY LIMITS ---
class biology:
    igf_std_limit = 2.0  
    pnp_std_limit = 2.0
    mad_threshold = 2.5 
    volatility_quantile = 0.95    
    male_cv_limit = 0.30          
    female_cv_limit = 0.50        

# --- 6. TARGET LABELS & CALIBRATION ---
class calibration:
    target_recall = 0.70 
    positive_label = "GH_CONTROL"
    unlabeled_label = "ATHLETE_REF"

# --- 7. DATASET COLUMN NAMING ---
class features:
    label_col = 'source'
    id_col = 'id'
    sex_col = 'sex'
    date_col = 'date'

# --- 8. VISUAL SHADES & PALETTES ---
class shades:
    # Consistent with your "Male Blue / Female Pink" islands
    C_BLUE   = '#1f77b4' 
    C_RED    = '#d62728' 
    C_PINK   = '#e377c2' 
    C_BLACK  = '#000000' 
    C_GREY   = '#7f7f7f' 
    C_GREEN  = '#27ae60' # Added for Fig 9 journey start points
    
    source_palette = {
        'ATHLETE_REF': '#1f77b4',
        'GH_CONTROL':  '#d62728',
        'GH_ADMIN':    '#000000',
        'POSITIVE':    '#d62728'
    }