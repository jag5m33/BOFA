## BOFA

BOFA: Bayesian Optimisation & Feature Analysis
1. PROJECT OVERVIEW
BOFA is a machine learning framework developed to detect anomalous athletes exhibiting Growth Hromone (GH)-like biomarker profiles. This project applies a consensus-based anomaly detection approach, combining multiple unserupvised models to identify athletes whose physiological signitures deviate from a learning baseline of normal biological vairation. 
The primary objective is to:

    Detect anomalous (GH-like) athletes from a reference populationusing a robust, mulit-model consensus framework.

BOFA addresses this by moving away from the commonly used static threshols (like the GH-2000 score) towards a consensus-based anomaly detection approach. The models learns the biological 'norm' with variation from a populaition a nd flags indiivudals who deviate into GH-synonymous physiological spaces.

2. REPO STRUCTURE
The repository is organised into modular segments to ensure reproducability and scalability, with a config file to prevent hard coding variables into the main-body of models. 
/prep   - contains congif.py (centralised parameter management to prevent hardcoding values into the main body script)

/models - machine learning models 
        - ae.py: Neural Network for dimensionality reduction and reconstruction error 
        - IF.py: Isolation Forest for parition based anomaly detection through scoring.
        - SVM.py: One-Class Support Vector Machine for boundary detection.
        - gmm.py: Gaussian Mixture Model for density-based probability estimation and scoring.

bofa_go.py: the primary execusion script which calls definitiions from scripts 

/results - Output CSVs includign flagged athletes and recall calibration logs

This structure ensures that data processing modelling and configuration steps are seperated and imporve maintainability. 

3. DATA PIPELINE 

    STEP 1: Preprocessing (RStudio v 4.5.1)

    Initial data preparation  is performed in R to ensure maximal harmonisation amongst datasets. THey originate from multiple sources with different assays:

        i. Log-Transformation of biomarkers (IGF-I, P-III-NP) to reduce biological skewed distribution 

        ii. Z-score Normalisation applied per assay column to address inter-assay vairability
        iii. Datasets were merged across multiple sources into a unified dataframe 

    STEP 2: Feature Engineerin agn Extraction (R)
        
        Following the preprocessing additional features were derived

        i. Aggregated biomarker features: 'avg_igf', 'avg_pnp' across assays 

        ii Categorical enconding: sex encoding as binary (0 = male, 1 = female)

        iii. Z-score normalisation of age

        iv. additional source column added: GH_CONTROL (for all gh_administed samples - 99), ATHLETE_REF (for all other remaining athletes to be assessed for doping based on bioamrker scores)

4. MODELLING PIPELINE

TRAINING STRATEGY:

    All models were trained exclusively on the ATHLETE_REF population which is reprsentative of athletes with unknown doping status. This establishes a baseline population model of athlete physiology.

    THis approach follows a one-class leanring paradigm where:

        - The model learns the distribution of the population data 

        - then deviations from this population distribution are flagged as anomalies (female model)

        - the male model uses the GH_CONTROL samples (which are all male) to perform a semi-supervised learning approach 

5. CONSENSUS FRAMEWORK

To imrpove robustness and reduce model specific biases, a consensus-based approach is applied

VOTING SYSTEM:

    a. each anomaly model assigns an anomaly label:

        - -1 : anomaly 

        - +1 : normal 

        - the vote strength is calculated as the total number of models that classify a sample as anomalous 

WEIGHTED SCORING:

    b. finalised anomaly score is calculated with a weighted combination of:

        - vote strength (primary signal)

        - individual model anoamly scores (refinement)

            THIS ENSURES:

                - Strong agreement between models * individual model confidence is considered for the athlete ranking 

6. EVALUATION METRICS

HOW WILL THE MODEL BE EVALUAED * CURRENT DEVELOPING 
            