from sklearn.mixture import GaussianMixture

# gmm: models the athlete population  as collection of 'clouds' (probability distributions) + athletes far from any cloud get high suspicion scores  
def get_gmm_scores(train_latent, full_latent, components, seed):
    model = GaussianMixture(n_components=components, random_state=seed)
    model.fit(train_latent)
    return -model.score_samples(full_latent)
# score_samples: calcualtes the denity  of an athelte; if it is in low_density are (no one else lives) = flagged
# # - models.decision_function: origonal score are negative (more negative = more anomalous) - we reverse it: - x - = + (so higher value = more anomalous)