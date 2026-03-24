# =================================================================
# SECTION 1: IMPORTS & CONFIG
# =================================================================
from pybofa.prep.config import model as mcfg, data as dcfg, isolation_forest as icfg, single_vector_machine as scfg
import pybofa.prep.vf_autoencoder as aut
import pybofa.models.IF as IF 
import pybofa.models.SVM as SVM 
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import random

# SET GLOBAL SEEDS (Do this before any other imports)
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# 2. Force TensorFlow to be deterministic (Optional but helps)
tf.config.experimental.enable_op_determinism()

# =================================================================
# SECTION 2: AUTOENCODER TRAINING & DATA SAVING
# =================================================================
# 1. Process Raw Data and load Metadata
df_tensor, n_features, ids, sources = aut.proc(dcfg.merged_df)
merged_raw = pd.read_csv(dcfg.merged_df)

# 2. Build and Train 3D Bottleneck
autoencoder = aut.autoenc_build(n_features, 3)
aut.compiler(autoencoder)
aut.train_autoencoder(autoencoder, data=df_tensor, epochs=mcfg.epochs)

# 3. Extract Raw Latent Space (Math ready)
encoder, _ = aut.latent_space(autoencoder)
latent_data_raw = aut.arrays(encoder, df_tensor) 

    # Look at how much the model struggled with each sample (identification and plotting wise)
reconstructions = autoencoder.predict(df_tensor) # predict the sample from the dimensioality reduction 
# find the average difference between the reconstruction and the true value. This looks at the MSE (loss of info.)
mse_loss = np.mean(np.square(df_tensor - reconstructions), axis =1)


# 4. CREATE & SAVE THE DATAFRAME + add  the error as a feature
latent_df = pd.DataFrame(latent_data_raw, columns=['z1', 'z2', 'z3'])
latent_df['recon_error'] = mse_loss
latent_df['id'] = ids.values
latent_df['source'] = sources.values
latent_df['sex'] = merged_raw['sex'].values # Attach gender directly

# Save to your folder so you can find it later
latent_df.to_csv('frozen_latent_data.csv', index=False)
print("--- SUCCESS: Coordinates saved to 'frozen_latent_data.csv' ---")

# =================================================================
# SECTION 3: GENDER-AWARE ANOMALY DETECTION
# =================================================================

#latent_df = pd.read_csv(dcfg.latent_df)

gh_samples = latent_df[latent_df['source'] == 'GH_CONTROL']
clean_athletes = latent_df[latent_df['source'] == 'ATHLETE_REF']

# --- 1. ISOLATION FOREST ---
features = ['z1', 'z2', 'z3', 'recon_error']

top_10_IF, processed_df, if_model = IF.find_anomalies(
    train_df=clean_athletes[features], 
    full_df=latent_df, 
    contam=icfg.contam, 
    estimators=icfg.estimators, 
    top=icfg.top
)

# Global Scale FIRST  = ensures that it is consistant across sex
# SVM - single global SVM
features = ['z1', 'z2', 'z3', 'recon_error']
scaler = StandardScaler()
scaler.fit(clean_athletes[features])

# FIT on clean data ONLY
train_scaled = scaler.fit_transform(clean_athletes[features])
full_scaled = scaler.transform(processed_df[features])

# Train the SVM
svm_model = OneClassSVM(kernel='rbf', gamma=scfg.gamma, nu=scfg.nu).fit(train_scaled)

# Apply to everyone
processed_df['svm_anomaly'] = svm_model.predict(full_scaled)

# Check if the GH samples actually look 'anomalous' to the Autoencoder
gh_samples = latent_df[latent_df['source'] == 'GH_CONTROL']
athlete_ref = latent_df[latent_df['source'] == 'ATHLETE_REF']

print(f"Athlete Mean Error: {athlete_ref['recon_error'].mean():.6f}")
print(f"GH Control Mean Error: {gh_samples['recon_error'].mean():.6f}")
# =================================================================
# SECTION 4: PERFORMANCE SUMMARY
# =================================================================
gh_total = len(gh_samples)
gh_caught_if = processed_df[(processed_df['source'] == 'GH_CONTROL') & (processed_df['anomaly'] == -1)]
gh_caught_svm = processed_df[(processed_df['source'] == 'GH_CONTROL') & (processed_df['svm_anomaly'] == -1)]

print("\n" + "="*40)
print("FINAL PERFORMANCE SUMMARY")
print("="*40)
print(f"ISOLATION FOREST RECALL: {len(gh_caught_if)}/{gh_total} ({len(gh_caught_if)/gh_total*100:.1f}%)")
print(f"SVM GENDER-SPLIT RECALL: {len(gh_caught_svm)}/{gh_total} ({len(gh_caught_svm)/gh_total*100:.1f}%)")
print("-" * 40)

# Final Visualization to see the Gender clusters
viz_scaler = StandardScaler()
scaled_viz_coords = viz_scaler.fit_transform(latent_df[['z1', 'z2', 'z3']])
IF.t_sne(scaled_viz_coords, latent_df['sex'].values)

