from sklearn.svm import OneClassSVM 
import pandas as pd

def one_svm(train_df, full_df, top, gamma, nu):
    
    x_train = train_df[['z1', 'z2', 'z3']].copy()
    x_full = full_df[['z1', 'z2', 'z3']].copy()

    # generate parameters for SVM:
        # rbf kernel best for determining 'boundary' of normal physiology
    svm = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu) 
    
    # TRAIN the mdoel on clean athletes ONLY - This teaches model the 'shape' of a clean athlete in latent space
    svm.fit(x_train)
    
    # PREDICT on the full dataset (Athletes + GH)
    full_df['svm_anomaly'] = svm.predict(x_full)
    full_df['svm_scores'] = svm.decision_function(x_full)

    # filter anomalies identified by SVM with  'source' column to find suspicious unknown athletes
        # by selecting anomalies (-1) which have source column: athlete ref: 
    athlete_anoms = full_df[(full_df['svm_anomaly'] == -1) & 
                            (full_df['source'] == 'ATHLETE_REF')]
    
    # sort by most extreme unknown cases first to less
 
    top_n = athlete_anoms.sort_values(by='svm_scores').head(top) 

    return svm, top_n, full_df