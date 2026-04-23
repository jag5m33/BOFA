from sklearn.semi_supervised import LabelSpreading
import numpy as np
# Import config
from pybofa.prep.config import ensemble_params as ecfg

def run_label_spreading(latent_space, labels):
    """
    Model 3: Graph-based Similarity (Label Spreading).
    Operates on the 3D latent manifold to propagate suspicion scores.
    """
    # 1. Prepare labels (established in main script)
    ls_labels = np.copy(labels)

    # Reference athletes (0) must become -1 = 'unlabeled' for propagation
    ls_labels[ls_labels == 0] = -1 
    
    # stop model from failing if no ground-truth positives are present (even though there are)
    if 1 not in ls_labels:
        print("[WARNING] No positive labels found for Label Spreading. Returning zero.")
        return np.zeros(len(ls_labels))

    # 2. build model of foundational 'normal' reference pop.
    # Uses ls_neighbors (3) and ls_alpha (0.1) from config.py
    model = LabelSpreading(
        kernel='knn', 
        n_neighbors=ecfg.ls_neighbors, 
        alpha=ecfg.ls_alpha # clamp factor: how much model trusts initial labels
    )
    
    # 3. Fit on the 3D latent space
    # This is where suspicion 'spreads' across the globular islands
    model.fit(latent_space, ls_labels)
    
    # 4. Extract Anomaly Probabilities
        # probability distribution for Class 1 (GH Positive)
    if model.label_distributions_.shape[1] > 1: # checks if there is the right amount of columsn 2
            #model.label_distributions = each row = athlete, col = class (0-norm, 1-doping) - scale of probabilities (they add up to 1 for each athlete)

        scores = model.label_distributions_[:, 1] # give me all row values column '2' (index 1) - doping 
    else:
        scores = np.zeros(len(ls_labels))
        
    return scores # all athletes in second columns which have scores of doping (high doping pattern = 0.98)