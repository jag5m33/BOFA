# BOFA
BOFA: Bayesian Optimization & Feature Analysis
Extended Research Project Github page for tracking all coding developments. This project will aim to create a machine learning model which is able to detect doping athletes from the population using the consensus approach.

To start with, there have been 3 main repository set up steps:
    1. Set up file system for the prep work 
    2. Set up file system for the latent space  
    3. Set up file system for the model development

3 main models will be used for the data being worked on:
    1. One class single vector machine (SVM) 
    2. isolation forest (if)
    3. gaussian mixture model (GMM)
They will all be developed on the foundation of a latent space created from an AUTOENCODER - this is an unsupervised neural network which compresses input data (Encoding) and then reconstructs it into a latent space representation (decoding)

This is a type of non-linear dimensionality reduction