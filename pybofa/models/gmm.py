from sklearn.mixture import GaussianMixture

# gmm: models the athlete population  as collection of 'clouds' (probability distributions) + athletes far from any cloud get high suspicion scores  
def run_gmm(train_data, full_data, cfg):
    model = GaussianMixture(
        n_components=cfg.gmm_components, 
        random_state=cfg.random_state,
        covariance_type='full'
    )
    model.fit(train_data)
    # Negate score_samples: lower log-likelihood = higher suspicion
    return model.score_samples(full_data)

# score_samples: calcualtes the denity  of an athelte; if it is in low_density are (no one else lives) = flagged
# # invert = true in the go script models.decision_function: origonal score are negative (more negative = more anomalous) - we reverse it: - x - = + (so higher value = more anomalous)



