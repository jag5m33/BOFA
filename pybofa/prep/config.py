
# section: image data
class data:
    merged_df = r'C:/Users/jagmeet/bofa_data/merged_df.csv'
    latent_space = r'C:/Users/jagmeet/bofa_data/model_results/athlete_latent_space.csv'
    latent_df = r'C:/Users/jagmeet/bofa_data/model_results/frozen_latent_data.csv'


# section: image processor
class processor:    
    validation_split = 0.2


#section: model
class model:
       #model fit
    epochs=50
    dim = 9

class isolation_forest:
    contam = 0.15      
    estimators = 250
    top = 10 
    components = 3
    init = 'random'
    learning_rate = 'auto'
    random_state = 42
    perplexity = 40
    n_iter = 1000
    max_samples = 64

class single_vector_machine:
    top = 10
    gamma = 'scale'
    nu = 0.15        
    





