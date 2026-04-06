import numpy as np
from sklearn.mixture import GaussianMixture

def run_gmm(train_data, full_data, doped_data, cfg):
    """
    Forensic Contrastive GMM:
    Calculates the log-likelihood ratio between a Reference population 
    and a Doped population. Includes scoring stabilization to prevent 
    unrealistic density spikes in dissertation plots.
    """
    
    # 1. Model the 'Normal' Biological Baseline (Reference Group)
    # n_components=2 allows the model to capture Male/Female sub-clusters
    gmm_ref = GaussianMixture(
        n_components=cfg.gmm_components, 
        covariance_type='full', 
        max_iter=200,
        random_state=cfg.random_state
    )
    gmm_ref.fit(train_data)
    
    # 2. Model the 'Doped' Distribution (Using GH_CONTROL samples)
    # A single component is usually enough to model the 'Doped' manifold
    gmm_doped = GaussianMixture(
        n_components=1, 
        covariance_type='full', 
        random_state=cfg.random_state
    )
    gmm_doped.fit(doped_data)
    
    # 3. Calculate Raw Log-Likelihoods
    log_prob_ref = gmm_ref.score_samples(full_data)
    log_prob_doped = gmm_doped.score_samples(full_data)
    
    # 4. Score Stabilization (Fixes the image_15cc41.jpg spike)
    # We clip extreme outliers to prevent the 'compressed' density plot effect
    log_prob_ref = np.clip(log_prob_ref, -50, None)
    log_prob_doped = np.clip(log_prob_doped, -50, None)
    
    # 5. Generate the Likelihood Ratio Score
    # Contrast = Log(P(Doped)) - Log(P(Ref))
    raw_contrast = log_prob_doped - log_prob_ref
    
    # Optional: Apply a log-transform to the contrast to spread the distribution
    # This ensures that the 'Logic Distribution' plot shows two clear, readable peaks
    final_scores = np.sign(raw_contrast) * np.log1p(np.abs(raw_contrast))
    
    return final_scores