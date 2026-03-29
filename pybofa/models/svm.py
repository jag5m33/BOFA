from sklearn.svm import OneClassSVM

#SVM: draws a 'boundary' around the dense cluster of normal athletes' any athltes outside the boundary gets a high suspicion score
def get_svm_scores(train_latent, full_latent, nu):
    model = OneClassSVM(kernel='rbf', nu=nu, gamma='auto')
    model.fit(train_latent)
    return -model.decision_function(full_latent)
# - models.decision_function: origonal score are negative (more negative = more anomalous) - we reverse it: - x - = + (so higher value = more anomalous)