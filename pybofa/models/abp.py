
def abp(df):
    """
    Calculates Intra-individual variance metrics (ABP style).
    Assumes sex-specific standardization has ALREADY occurred in load_and_preprocess.
    """
    # 1. Filter for athletes with more than 2 data samples (to ensure meaningful variance calculation) 
    # This ensures we have enough points to calculate a meaningful Standard Deviation
    df_long = df.groupby('id').filter(lambda x: len(x) > 1).copy()
    
    # 2. Define the markers to track
    # Using the columns your load_and_preprocess just standardized
    markers = ['avg_pnp', 'avg_igf', 'igf_pnp_ratio']
    
    # 3. Aggregate: Calculate the 'Personal Baseline' (Mean) and 'Volatility' (SD)
    features = df_long.groupby(['id', 'sex']).agg({
        m: ['mean', 'std', 'max'] for m in markers
    })
        # lambda (x) creates a function which isolates the features of id's (indexes) that have more than 3 samples
        
    
    # 4. Flatten the multi-level headers created by .agg()
    features.columns = ['_'.join(col) for col in features.columns]
    
    # 5. Calculate Coefficient of Variation (CV)
    # This is the 'Suspicion' metric: How much does this athlete swing?
    for m in markers:
        # add  epsilon (1e-6) to avoid division by zero
        features[f'{m}_cv'] = features[f'{m}_std'] / (features[f'{m}_mean'] + 1e-6)
        
    return features.reset_index()
        # returns the id (which becomes the 'index' = row of labels)
        # moves ids back to regular column. easy for plotting 
