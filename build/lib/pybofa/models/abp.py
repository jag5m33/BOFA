def abp(df):
    """
    Model 0: Athlete biological passport.
    outlier detection based on longitudanal approaches
    """
    markers = ['avg_pnp', 'avg_igf', 'igf_pnp_ratio']
    
    # 1. count and identify athlete IDs that have 2 or more samples
    counts = df['id'].value_counts()
    multi_sample_ids = counts[counts >= 2].index
    
    # 2. Filter df for only those IDs
    passport_df = df[df['id'].isin(multi_sample_ids)]
    
    # 3. Calculate metrics
    features = passport_df.groupby(['id', 'sex']).agg({
        m: ['mean', 'std', 'max'] for m in markers
    })
    
    # Flatten columns and rename std to volatility
    features.columns = ['_'.join(col) for col in features.columns]
    for m in markers:
        features[f'{m}_volatility'] = features[f'{m}_std']
        # just renaming and ensured that the columns are mean volatility and max
    return features.reset_index()