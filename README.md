# BOFA

BOFA: Bayesian Optimization & Feature Analysis
Extended Research Project Github page for tracking all coding developments. This project will aim to create a machine learning model which is able to detect doping athletes from the population using the consensus approach.

To start with, there have been 3 main repository set up steps:
    1. Set up file system for the prep work 
    2. Set up file system for the latent space  
        - compute latent space GH scoring vector - this will be used later to determine whether flagged athletes are close or far from GH projection in Latent space (consensus stage)
    3. Set up file system for the model development

3 main models will be used for the data being worked on:
    1. One class single vector machine (SVM) 
    2. isolation forest (if)
    3. gaussian mixture model (GMM)
They will all be developed on the foundation of a latent space created from an AUTOENCODER - this is an unsupervised neural network which compresses input data (Encoding) and then reconstructs it into a latent space representation (decoding)

This is a type of non-linear dimensionality reduction
A config and TOML file will also be used to prevent any hard coded information to be included into the model and to keep track of the versions of packages and models used.

1st step is pre-processing' on the GHadmin. dataset (control) and the unlabelled data separately:

    1. Log-transform markers (IGF-I and P-III-NP) to handle biological skewness.

    2. Z-score normalise markers (separately for each assay: CIS, Orion, etc.).
    
    3. Z-score normalise Age.
    
    4. Encode sex (0 for Male, 1 for Female).
    
    5. Aggregate: Group by Volunteer and calculate the mean of the Z-scores to get your final row: [AthleteID, Age_Z, sex, IGF_Z_Avg, PIIIP_Z_Avg].

