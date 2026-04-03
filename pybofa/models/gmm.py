import numpy as np
from sklearn.mixture import GaussianMixture

def run_gmm(train_data, full_data, doped_data, cfg):
    # Pure Unsupervised/Semi-supervised approach:
    # Train on reference, score the full set.
    gmm = GaussianMixture(n_components=cfg.gmm_components, 
                          covariance_type='full', 
                          random_state=cfg.random_state)
    gmm.fit(train_data)
    # Higher score should mean 'less likely to be clean'
    return -gmm.score_samples(full_data)