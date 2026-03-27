from pybofa.prep.config import model as mcfg
from pybofa.prep.config import data as dcfg
from pybofa.prep.config import isolation_forest as icfg
from pybofa.prep.config import single_vector_machine as scfg
from pybofa.prep.config import gauss as gcfg
from pybofa.prep.config import processor as pcfg
from pybofa.prep.config import conclusion_metrics as ccfg

import pybofa.prep.vf_autoencoder as aut
import pybofa.models.IF as IF 
import pybofa.models.SVM as SVM 
import pybofa.models.gmm as gmm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import tensorflow as tf
import random

# ================================================================
# GLOBAL SEEDS (REPRODUCIBILITY)
# ================================================================
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
tf.config.experimental.enable_op_determinism()

features = pcfg.features

# ================================================================
# SECTION 1: DATA PROCESSING
# ================================================================
merged_raw = pd.read_csv(dcfg.merged_df)

df_tensor, n_features, ids, sources = aut.proc(dcfg.merged_df)

athlete_mask = (sources.values == 'ATHLETE_REF')
athlete_tensor = df_tensor[athlete_mask]   # FIXED (previously missing)

# ================================================================
# SECTION 2: AUTOENCODER TRAINING (NO DATA LEAKAGE)
# ================================================================
train_data = athlete_tensor

autoencoder = aut.autoenc_build(n_features, 5)
aut.compiler(autoencoder)

aut.train_autoencoder(
    autoencoder,
    data= train_data,
    epochs=mcfg.epochs
)

# Reconstruction error (computed on FULL dataset)
reconstructions = autoencoder.predict(df_tensor)
recon_error_matrix = np.square(df_tensor - reconstructions)

mse_loss = recon_error_matrix.mean(axis=1)
max_loss = recon_error_matrix.max(axis=1)

# Combine both (better signal)
mse_loss = 0.7 * mse_loss + 0.3 * max_loss

# Latent space
encoder, _ = aut.latent_space(autoencoder)
latent_data_raw = aut.arrays(encoder, df_tensor)

# elbow check
dims, errors = aut.elbow_plot(n_features, 9, athlete_tensor)
aut.plotting(dims, errors)

# ================================================================
# SECTION 3: CREATE LATENT DATAFRAME
# ================================================================
latent_df = pd.DataFrame(latent_data_raw, columns=features)

latent_df['recon_error'] = mse_loss
latent_df['id'] = ids.values
latent_df['source'] = sources.values
latent_df['sex'] = merged_raw['sex'].values

latent_df.to_csv('frozen_latent_data.csv', index=False)

print("--- SUCCESS: Coordinates saved ---")

# ================================================================
# SECTION 4: ANOMALY MODELS
# ================================================================
gh_samples = latent_df[latent_df['source'] == 'GH_CONTROL']
clean_athletes = latent_df[latent_df['source'] == 'ATHLETE_REF']

# -----------------------------
# Isolation Forest
# -----------------------------
top_10_IF, processed_df, if_model = IF.find_anomalies(
    train_df=clean_athletes[features],
    full_df=latent_df,
    contam=icfg.contam,
    estimators=icfg.estimators,
    top=icfg.top
)

# -----------------------------
# SVM
# -----------------------------
svm_model, top_10_svm, processed_df = SVM.one_svm(
    train_df=clean_athletes,
    full_df=processed_df,
    top=scfg.top,
    gamma=scfg.gamma,
    nu=scfg.nu
)

# -----------------------------
# GMM
# -----------------------------
gmm_thresh, gmm_flags = gmm.gmm(
    train_df=clean_athletes,
    full_df=processed_df,
    contamination=gcfg.contamination,
    n_components=gcfg.n_components,
    random_state=gcfg.random_state
)

processed_df['gmm_anomaly'] = gmm_flags

# ================================================================
# SECTION 5: CONSTRUCTING SCORING TABLE
# ================================================================

athlete_mask = processed_df['source'] == 'ATHLETE_REF'
gh_mask = processed_df['source'] == 'GH_CONTROL'

# -----------------------------
# Reconstruction Z-score (NO LEAKAGE)
# -----------------------------
ath_recon = processed_df.loc[athlete_mask, 'recon_error']

mean_recon = ath_recon.mean()
std_recon = ath_recon.std()

processed_df['recon_z'] = (processed_df['recon_error'] - mean_recon) / std_recon
processed_df['recon_z_clipped'] = processed_df['recon_z'].clip(-3, 4)

# -----------------------------
# Voting system
# -----------------------------
processed_df['vote_strength'] = (
    (processed_df['anomaly'] == -1).astype(int) +
    (processed_df['svm_anomaly'] == -1).astype(int) +
    (processed_df['gmm_anomaly'] == -1).astype(int)
)

processed_df['vote_score'] = np.sqrt(processed_df['vote_strength'])

# -----------------------------
# Final score
# -----------------------------
processed_df['final_score'] = (
    processed_df['recon_z_clipped'] * 0.7 +
    processed_df['vote_score'] * 0.3
)

processed_df['moderate_bonus'] = (
    (processed_df['recon_z_clipped'] >= 2) &
    (processed_df['recon_z_clipped'] <= 4)
).astype(int)

processed_df['final_score'] += 0.5 * processed_df['moderate_bonus']

# ================================================================
# SECTION 6: THRESHOLDING
# ================================================================
threshold = np.percentile(
    processed_df.loc[athlete_mask, 'final_score'],
    ccfg.recon_perc
)

processed_df['final_flag'] = processed_df['final_score'] > threshold

# ================================================================
# SECTION 7: REPORTING
# ================================================================
def report_stats(mask, label):
    subset = processed_df[mask]
    detected = subset['final_flag'].sum()
    total = len(subset)
    print(f"{label}: {detected}/{total} ({detected/total*100:.2f}%)")

print("\nFINAL PERFORMANCE")
report_stats(gh_mask, 'Recall (GH)')
report_stats(athlete_mask, 'False Positives')

# ================================================================
# SECTION 8: MODEL COMPARISON
# ================================================================
metrics = ['anomaly', 'svm_anomaly', 'gmm_anomaly', 'final_flag']
summary = []

for col in metrics:
    if col == 'final_flag':
        gh_hits = processed_df[gh_mask][col].sum()
        ath_fps = processed_df[athlete_mask][col].sum()
    else:
        gh_hits = (processed_df[gh_mask][col] == -1).sum()
        ath_fps = (processed_df[athlete_mask][col] == -1).sum()

    summary.append({
        'Model': col,
        'Recall (GH)': f"{gh_hits}/{gh_mask.sum()}",
        'False Positives': f"{ath_fps}/{athlete_mask.sum()}"
    })

print(pd.DataFrame(summary))

# ================================================================
# SECTION 9: TOP ANOMALIES
# ================================================================
print(
    processed_df
    .sort_values('final_score', ascending=False)
    [['id', 'source', 'recon_error', 'recon_z', 'vote_score', 'final_score']]
    .head(10)
)

# ================================================================
# SECTION 10: VISUALISATION (SEPARATE SCALING)
# ================================================================
viz_scaler = StandardScaler()

# IMPORTANT: fit on clean data only (avoids leakage)
viz_scaler.fit(clean_athletes[features])

scaled_viz_coords = viz_scaler.transform(processed_df[features])

IF.t_sne(scaled_viz_coords, processed_df['sex'].values)