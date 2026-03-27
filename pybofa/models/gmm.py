
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from numpy import quantile
import matplotlib.pyplot as plt
from pybofa.prep.config import gauss as gcfg
import numpy as np
from pybofa.prep.config import processor as pcfg

def gmm(train_df, full_df, features, contamination, n_components, random_state):
    x_train = train_df[features].copy()
    x_full = full_df[features].copy()
    gausmix = GaussianMixture(n_components= n_components,  random_state = random_state).fit(x_train)

    full_scores = gausmix.score_samples(x_full)
    train_scores = gausmix.score_samples(x_train)

    thresh = quantile(train_scores, contamination) # extract the thresold for anomaly deteciton using quantile 

    # Return flags: -1 for anomaly, 1 for normal (to match IF/SVM style)
    flags = np.where(full_scores <= thresh, -1, 1)
    return thresh, flags, gausmix

# 