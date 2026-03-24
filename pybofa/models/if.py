import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import pandas as pd
from pybofa.prep.config import isolation_forest as icfg
from sklearn.manifold import TSNE

def t_sne(latent_data, labels):

    # use t-SNE for 3D visualization (elbow plot shows 3 dimensions is the right amount to capture reduced info.)
    tsne = TSNE(
        n_components=icfg.components, 
        random_state=icfg.random_state, 
        perplexity=icfg.perplexity, 
        learning_rate=icfg.learning_rate, 
        init=icfg.init, 
        max_iter=icfg.n_iter 
    )
    x_tsne = tsne.fit_transform(latent_data)

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(x_tsne[:, 0], 
                         x_tsne[:, 1], 
                         x_tsne[:, 2], 
                         c=labels, 
                         cmap='coolwarm', 
                         alpha=0.6, 
                         s=2)

    ax.set_title("3D t-SNE: Distribution by Gender")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_zlabel("t-SNE 3")
    
    # legend 
    cbar = plt.colorbar(scatter, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Male (0)', 'Female (1)'])
    
    plt.show()
    
    return x_tsne


def find_anomalies(train_df, full_df, contam, estimators, top):
    # Use the coordinates for training/prediction (training - unseen data, validation (control))
    features = ['z1', 'z2', 'z3', 'recon_error']
    x_train = train_df[features].copy()
    x_full = full_df[features].copy()

    train_df['recon_error'] = train_df['recon_error'] * 100
    full_df['recon_error'] = full_df['recon_error'] * 100

    # Initialize Forest
    iso_forest = IsolationForest(
        n_estimators=estimators,
        random_state=icfg.random_state, 
        contamination=contam,
        max_samples=icfg.max_samples, 
        bootstrap=True
    )
    
    # Train on clean athletes only to establish "normal" boundaries
    iso_forest.fit(x_train)
    
    # Predict on everything
    full_df['anomaly'] = iso_forest.predict(x_full)
    # decision_function gives lower (more negative) scores to more isolated points
    full_df['scores'] = iso_forest.decision_function(x_full) 
     
    #Use 'source' column from original df to identify unknown athlete anomalies
    #  athletes flagged (-1) &  NOT in the GH control group
    athlete_anoms = full_df[(full_df['anomaly'] == -1) & 
                            (full_df['source'] == 'ATHLETE_REF')] 
    
    # Sort by score  --> get highest (most extreme) first
    top_n = athlete_anoms.sort_values(by='scores').head(top) 

    return top_n, full_df, iso_forest