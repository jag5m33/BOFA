from pybofa.prep.config import data as dcfg, processor as pcfg, isolation_forest as icfg, single_vector_machine as scfg, gauss as gcfg, conclusion_metrics as ccfg
from pybofa.models import IF as IF, SVM as SVM, gmm as gmm 
import pybofa.models.SVM as SVM 
import pybofa.models.gmm as gmm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
# THIS SCRIPT RUNS PURELY ON MEN AS THE CONTROL DATA IS ALL MALE DATA.


# load master dataset
merged_df = pd.read_csv(dcfg.merged_df)
merged_df['sex'] = merged_df['sex'].astype(int)
male_df = merged_df[merged_df['sex'] == 0]

# then impute to remove the NA and only use valid rows 
imputer = SimpleImputer(strategy= 'median')
male_df[pcfg.features] = pd.DataFrame(
    imputer.fit_transform(male_df[pcfg.features]),
    columns = pcfg.features,
    index = male_df.index
)
#create 2 datasets: athlete_ref and gh_control 
athlete_mask = male_df['source'] == 'ATHLETE_REF'
gh_mask = male_df['source'] == 'GH_CONTROL'



# scaling + no need for it since scaling was already performed in R 
#scaler = StandardScaler()
#merged_df[pcfg.features] = scaler.fit_transform(merged_df[pcfg.features])

# create dataset: clean athletes with set columns of features from athlete_ref source
clean_athletes = male_df.loc[athlete_mask, pcfg.features] 


#train IF
top_10_IF, processed_df, if_model = IF.find_anomalies(
    train_df = clean_athletes,
    full_df= male_df.copy(),
    features = pcfg.features,
    contam = icfg.contam,
    estimators= icfg.estimators,
    top = icfg.top
)

print("finished training Isolation Forest")
processed_df['if_score'] = -if_model.decision_function(processed_df[pcfg.features])
#train one class SVM
svm_model, top_10_SVM, processed_df = SVM.one_svm(
    train_df = clean_athletes,
    full_df=processed_df,
    features = pcfg.features,
    top = scfg.top,
    gamma = scfg.gamma,
    nu = scfg.nu
)

print("finished training One class Single Vector Machine")
processed_df['svm_score'] = -svm_model.decision_function(processed_df[pcfg.features])
#train gmm
gmm_thresh, gmm_flags, gmm_model = gmm.gmm(
    train_df = clean_athletes, 
    full_df= processed_df,
    features = pcfg.features,
    contamination= gcfg.contamination,
    n_components= gcfg.n_components,
    random_state= gcfg.random_state
)
processed_df['gmm_anomaly'] = gmm_flags

print("finished training Gaussian Mixture Model")
processed_df['gmm_score'] = -gmm_model.score_samples(processed_df[pcfg.features])

#standardise scores of each model anomaly selections:
for col in ['if_score', 'svm_score', 'gmm_score']:
    processed_df[col] = (
        (processed_df[col] - processed_df[col].mean()) /
        processed_df[col].std()
    )

#voting system: vote_Strength is an integer count of how many models flagged sample
    # if samples get flagged by all 3 models, then they will all have the same final score (sqrt(3) - from line below voting system)

processed_df['vote_strength'] = (
    (processed_df['if_anomaly'] == -1).astype(int)+
    (processed_df['svm_anomaly'] == -1).astype(int)+
    (processed_df['gmm_anomaly'] == -1).astype(int)
)
#use sqrt to remove the effect of higher values
processed_df['vote_score'] = np.sqrt(processed_df['vote_strength']) 

# since no dimensionality reduction is used use vote_score to be a main anomaly indicator (measure)
    # most weight on votes, but also reanked with the if score , svm score adn gmm scores
processed_df['final_score'] = (
    0.5 * processed_df['vote_strength'] +   # keep consensus strong
    0.2 * processed_df['if_score'] +
    0.15 * processed_df['svm_score'] +
    0.15 * processed_df['gmm_score']
)

#threshold the top n% of athletes:
# isolate the score sof athlete_ref group based on their final_score  - caculates the cut off to be the n%
threshold = np.percentile(
    processed_df.loc[athlete_mask, 'final_score'],
    ccfg.recon_perc
)

processed_df['final_flag'] = processed_df['final_score'] > threshold # all the athleres above the threshold flag them in anotehr columns with final_flag

#report the results:
def report_stats(mask, label):
    subset = processed_df[mask]
    detected = subset['final_flag'].sum() # add up all athletes from the mask subset of processed_df who were detected by any of the models
    total = len(subset) # what is the total of all flags across athletes
    print(f"{label}: {detected}/{total} ({detected/total*100:.2f}%)")
print("\n final performance")
report_stats(gh_mask, 'recall(GH)')
report_stats(athlete_mask, 'false positives')

#top anomalies
print(processed_df[athlete_mask]
      .sort_values('final_score', ascending=False)
      [['id', 'source', 'final_score']]
      .head(10)
    )

print(pd.crosstab(male_df['source'], male_df['sex']))