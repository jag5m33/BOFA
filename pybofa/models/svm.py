from sklearn.svm import OneClassSVM

def get_svm_scores(train_latent, full_latent, nu):
    model = OneClassSVM(kernel='rbf', nu=nu, gamma='auto')
    model.fit(train_latent)
    return -model.decision_function(full_latent)