#if - isolation forest
# the standard packages for IF is sicikit learn - 
    # this requires the tensor dataset to be turned into  numpy array or pandas dataframe to feed into forest
from pybofa.prep.config import model as mcfg
from pybofa.prep.config import data as dcfg
from pybofa.prep.config import isolation_forest as icfg

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample 

def t_sne(latent_data):
    
    tsne = TSNE(n_components = 3, random_state = 42, perplexity = 30)
    x_tsne = tsne.fit_transform(latent_data)

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection = '3d')
    scatter = ax.scatter(x_tsne[:, 0], x_tsne[:, 1], x_tsne[:, 2], cmap = 'viridis', s = 2)

    ax.set_title("3D t-SNE visualisation of autoencoder-encoder latent space co-ordinates - dimensionality reduction visual")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_zlabel("t-SNE 3")
    plt.colorbar(scatter, label = 'classes')
    plt.show()
    return latent_data

def find_anomalies(df, contam, estimators, top):
    x = df[['z1', 'z2', 'z3']].copy()

    iso_forest = IsolationForest(n_estimators = estimators,
                                 random_state = 42, 
                                 contamination= contam)
    # do this ON x from DF becuase DF still has the ID column x is just the vector coords
    df['anomaly'] = iso_forest.fit_predict(x)

    df['scores'] = iso_forest.decision_function(x) # give lower scores to more isolated points
     

    anomalies = df[df['anomaly'] == -1].sort_values(by = 'scores') # anoamlies are the ones flagged with -1
    top_10 = anomalies.head(top) # take out the top 10 samples 

    print(f"number of anomalous samples: {len(anomalies)}, out of {len(df)} athletes")

    return top_10, iso_forest





