def abp(df):
    """
    Calculates Intra-individual variance metrics (ABP style).
    FIX: Uses Standard Deviation instead of CV for standardized data 
    to prevent 'exploding' suspicion scores.
    """
    # 1. Filter for athletes with longitudinal data
    df_long = df.groupby('id').filter(lambda x: len(x) > 1).copy()
    
    # 2. Markers to track (using the columns standardized in load_and_preprocess)
    markers = ['avg_pnp', 'avg_igf', 'igf_pnp_ratio']
    
    # 3. Aggregate: Calculate Baseline and Volatility
    # Use 'std' as the primary suspicion metric for normalized data
    features = df_long.groupby(['id', 'sex']).agg({
        m: ['mean', 'std', 'max'] for m in markers
    })
    
    # 4. Flatten columns
    features.columns = ['_'.join(col) for col in features.columns]
    
    # 5. Define the 'Volatility' metric 
    # For Z-scored data, 'std' is a better proxy for biological instability than CV
    for m in markers:
        features[f'{m}_volatility'] = features[f'{m}_std']
        
    return features.reset_index()