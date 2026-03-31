from sklearn.svm import OneClassSVM

#SVM: draws a 'boundary' around the dense cluster of normal athletes' any athltes outside the boundary gets a high suspicion score
def run_svm(train_data, full_data, cfg):
    model = OneClassSVM(
        kernel='rbf', 
        nu=cfg.svm_nu, 
        gamma='auto'
    )
    model.fit(train_data)
    # Negate decision function so higher values represent anomalies
    return model.decision_function(full_data)
# # invert = true in the go script models.decision_function: origonal score are negative (more negative = more anomalous) - we reverse it: - x - = + (so higher value = more anomalous)