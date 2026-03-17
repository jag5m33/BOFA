#1. Log transform (essential)
    # log-transform raw conc. values - the marker conc. are usually log-normal so to stabilise the vairance and ensure that Z-score normalisation is not biased by a few very high spikes in conc. 
    # value_log = ln(value_Raw)
#2. Assay SPECIFIC z-score normalisation 
    #normalise within each assay column before you combine them - DO NOT CIS AND ORION in same calculation 
    # for each measurment z = (val_log - mean_assay) / sd_assay
    # calculate the mean and SD for all CIS values use those to Z score the CIS column then calcualte a speerate mean and SD for all orion values and Z score that column
#3. Averaging - CSV has missing data - some smaples are not processed by both methods - this is fine, just use that value, but for those that are, calculate average  

#4. Aggregate per athlete - one you have a single Z score for P-III-NP and a single z score for IGF-I for every time point - you can group by the volunteer ID.

# one hot encode sex as binary (0/1)s

# age levels MUST be normalised 

import os 
import pandas as pd
from config import data as dcfg
import numpy as np

df = pd.read_csv(dcfg.gh_admin)




