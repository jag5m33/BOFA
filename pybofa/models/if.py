from sklearn.ensemble import IsolationForest

# The way an IF works: randomly splits data  + anomalies are lonely points that easy to isolate with very few splits 

def run_iforest(train_data, full_data, cfg):
    model = IsolationForest(
        contamination=cfg.iforest_contam, 
        random_state=cfg.random_state,
        n_estimators=200
    )
    model.fit(train_data)
    # Negate the decision function: more negative = more anomalous
    return model.decision_function(full_data)
# decision function: returns a score of how easy it was to isolate a point
# # invert = true in the go script models.decision_function: origonal score are negative (more negative = more anomalous) - we reverse it: - x - = + (so higher value = more anomalous)