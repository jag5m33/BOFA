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

*a key step here is to also set up a config file, this will prevent any hard coded variables throughout the code; filepathways, dimensions numbers, column numbers etc. these will all be specified in the config file and then called in the subsequent files for autoenc., models, or go.

1st step is pre-processing' on the GHadmin. dataset (control) and the unlabelled data separately:

    1. Log-transform markers (IGF-I and P-III-NP) to handle biological skewness.

    2. Z-score normalise markers (separately for each assay: CIS, Orion, etc.).
    
    3. Z-score normalise Age.
    
    4. Encode sex (0 for Male, 1 for Female).
    
    5. Aggregate: Group by Volunteer and calculate the mean of the Z-scores to get your final row: [AthleteID, Age_Z, sex, IGF_Z_Avg, PIIIP_Z_Avg].

    5. add in an extra methods column to distinguish between endocrine and biomarker samples and ensure that they specify whether hameolysis occurs or not 


2nd Step: Setup the autoencoder and generate a latent space matrix for all athlete samples

    1. building the autoencoder required specific packges: tensorflow.keras, pandas, matplotlib (for the elbow plot and t-sne)
    2. since the merged_df is going to be imported, specify a file pathway in the config file and call it into this file &  remove the first column (0) and the sixth column (5) as they are id's and method details. This means that they do not get processed as numerical values in the matrix that the autoencoder uses.
       
        a. ensure that the rows are not shuffled, otherwise we don't know which athlete belongs to which vector numerical matrix.
    
    2. convert to tensor return the numnber of features (columns), rows,  and build the mode layer by layer 

        a. ensure that at the smalled point of the first half of the autoencoder script (the encoder) the layer is labeled = bottleneck - this is important for the t-sne downstream of this model
    
    3. after building both the encoder and decoder = autoencoder, compile, train, and generate an elbow plot

        a. the elbow plot will explain the number of dimensions which has the lowest trade off between MSE and the predictions that come out of the autoencoder 

        b. remember the autoencoder is a type of predictive unseen neural network, so the encoder is performing dimensionality reduction, 
        the decoder is performing the rebuild of the data that was input and reduced. 
        if the model learned well, the reduction vectors are accurate from the bottlenck layer and the decoder output data will be very similar to the input data.

    4. from the bottleneck layer which was previously labelled, you want to run the autoencoder again but just the encoder part so that it stops at the bottlenck layer and extracts the most compressed vectors of the data - these are what build our latnet space 

    5. from there we can put the vectors into a numpy array ready for downstream GMM, IF, SVM to use, since they don't use tensor format.

3rd Step: Building the isolation forest. This model is based on paritioning. It will take the vectors matrix and start performing partitions in the shape of a tree. The faster the partition finishes (ie. no more splits can occur) the more likely the ending samples are erronous (they dont have surrounding data to further partition into)

    - this is how anomalies are found with this model.

    1. 

