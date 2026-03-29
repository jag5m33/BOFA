from sklearn.ensemble import IsolationForest

def get_iforest_scores(train_latent, full_latent, contam, seed):
    model = IsolationForest(contamination=contam, random_state=seed)
    model.fit(train_latent)
    return -model.decision_function(full_latent)