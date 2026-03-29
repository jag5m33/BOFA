from sklearn.mixture import GaussianMixture

def get_gmm_scores(train_latent, full_latent, components, seed):
    model = GaussianMixture(n_components=components, random_state=seed)
    model.fit(train_latent)
    return -model.score_samples(full_latent)