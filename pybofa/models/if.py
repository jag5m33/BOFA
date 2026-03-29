from sklearn.ensemble import IsolationForest

# The way an IF works: randomly splits data  + anomalies are lonely points that easy to isolate with very few splits 
def get_iforest_scores(train_latent, full_latent, contam, seed):
    model = IsolationForest(contamination=contam, random_state=seed)
    model.fit(train_latent)
    return -model.decision_function(full_latent) 
# decision function: returns a score of how easy it was to isolate a point