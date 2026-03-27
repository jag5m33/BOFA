
# section: image data
class data:
    merged_df = r'C:/Users/jagmeet/bofa_data/merged_df.csv'
    latent_space = r'C:/Users/jagmeet/bofa_data/model_results/athlete_latent_space.csv'
    latent_df = r'C:/Users/jagmeet/bofa_data/model_results/frozen_latent_data.csv'


# section: image processor
class processor:    
    validation_split = 0.2
    features = [
    'age', 'sex',
    'ln_pnp_orion', 'ln_pnp_cis', 'ln_pnp_siemens',
    'ln_pnp_initial', 'ln_pnp_mean',
    'ln_igf_immuno', 'ln_igf_immulite',
    'ln_igf_ms', 'ln_igf_ids', 'ln_igf_imt',
    'ln_igf_initial', 'ln_igf_mean',
    'avg_pnp', 'avg_igf'
    ]
    
#section: model
class model:
       #model fit
    epochs=100
    dim = 9
    patience = 5

class isolation_forest:
    contam = 0.05 #(sklearn default)    
    estimators = 500
    max_samples = 512
    random_state = 42
    
    top = 10 
    components = 3
    init = 'random'
    learning_rate = 'auto'
    
    perplexity = 40
    n_iter = 1000
    

class single_vector_machine:
    top = 10
    gamma = 'auto' # instead of scale - to prevent overfitting 
    nu = 0.05 # analagous to contamination for other two models


class gauss:
    #threshold = .03
    contamination =  0.05
    n_components = 4
    random_state = 42

class conclusion_metrics:
    recon_perc = 90






