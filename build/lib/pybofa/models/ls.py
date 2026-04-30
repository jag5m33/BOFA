from sklearn.semi_supervised import LabelSpreading
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import rankdata # Needed for the percentile gradient fix
# Import config
from pybofa.prep.config import ensemble_params as ecfg , model_params as mcfg

def run_label_spreading(latent_space, labels):
    """
    Model 3: Graph-based Similarity (Label Spreading).
    Operates on the 3D latent manifold to propagate suspicion scores.
    Uses an internal anchoring trick to stop the probability collapse.
    """
    # 1. prep labels (from main script)
    # make a copy since orig. labels are Read only. 
    ls_labels = np.copy(labels)


    # establish model - normal/ 'ref' --> find reference atheltes = label 0).
    ref_indices = np.where(labels == 0)[0]
    
    # We set a seed so the 'random' choice is the same every time we run the dissertation code
    np.random.seed(mcfg.random_state) 
    
    # VAL SPLITS
    # 30% of the refs as known '0' (anchors).
    # 70% become -1 = 'unlabeled' 

    # Gives anomalous atheltes a 'Clean' baseline to deviate from
    to_unlabel = np.random.choice(
        ref_indices, 
        size=int(len(ref_indices) * 0.7), 
        replace=False
    )
    ls_labels[to_unlabel] = -1 
    
    # prevent model from failing if no ground-truth positives are present (even though there are)
    
    if 1 not in ls_labels:
        print("No positive labels found for Label Spreading. return zero.")
        return np.zeros(len(ls_labels))
    scaler = StandardScaler()
    scaled_latent = scaler.fit_transform(latent_space)
    # 2. build model using rbf kernel to look at global 3D t-sne distribution
        # RBF kernel = 'heat map synonymous'; it assigns suspicion based on 3D distance.
    # Uses ls_gamma (0.3) and ls_alpha (0.1) from config.py
    
    model = LabelSpreading(
        kernel='knn', 
        alpha=ecfg.ls_alpha,
        n_neighbors = ecfg.ls_neighbours  
    )
    
    # 3. Fit on the 3D latent space
    # This is where suspicion 'spreads' across the clusters
    model.fit(scaled_latent, ls_labels)
    
    # 4. Extract Anomaly Probabilities
    # probability distribution for Class 1 (GH Positive)
    if model.label_distributions_.shape[1] > 1: # checks if there is the right amount of columns (2)
        # raw_probs: each row = athlete, col = class (0-norm, 1-doping)
        # probabilities add up to 1 for each athlete
        raw_probs = model.label_distributions_[:, 1] # give me all row values column '2' (index 1) - doping 
        

        # Convert binary probabilities into a smooth range for the PR and KDE plots
        # This ensures every athlete gets a unique ranking based on their 'heat' in the 3D space.
        # This stops the PR curve from being a flat diagonal line.
        scores = rankdata(raw_probs, method='average') / len(raw_probs)
        
    else:
        # Fallback if the model can't distinguish classes
        scores = np.zeros(len(ls_labels))
        
    return scores # all athletes ranked by doping pattern (high doping rank = 0.99)