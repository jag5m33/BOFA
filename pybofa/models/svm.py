from sklearn import svm
import numpy as np
from pybofa.prep.config import ensemble_params as ecfg

def run_svm(train_latent, train_labels, test_latent):
    """
    Model 2: Weighted Support Vector Machine (SVM).
    Operates on the 3D latent manifold to detect geometric anomalies.
    """
    
    # Weighted SVM: missing a GH sample is heavily penalized based on config
    # We use svm_weight from config to define the penalty for the positive class (1)
    clf = svm.SVC(
        kernel='rbf', 
        C=ecfg.svm_C,             # Regularization parameter from config
        gamma=ecfg.svm_gamma,    # Boundary smoothness from config
        probability=True, 
        class_weight={0: 1, 1: ecfg.svm_weight} # Penalty ratio from config
    )
    
    # Fit the model on the latent coordinates provided by the SSAE
    clf.fit(train_latent, train_labels)
    
    # Return the decision function (signed distance from the boundary)
    # Positive values indicate the anomaly zone; negative values indicate the baseline zone.
    scores = clf.decision_function(test_latent)
    
    return scores