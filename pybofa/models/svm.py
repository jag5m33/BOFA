from sklearn.svm import SVC

def run_svm(X_train, y_train, X_full):
    # We weight the 99 doped samples heavily to balance against 6000 clean ones
    model = SVC(kernel='rbf', probability=True, class_weight={1: 50, 0: 1})
    
    model.fit(X_train, y_train)

    # Return the probability of being in Class 1 (Doped)
    return model.predict_proba(X_full)[:, 1]