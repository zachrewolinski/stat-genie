from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/add_features_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe suitable for analysis.

    Produces:
      - ChoseDemonstrated: binary (1 if y==2 or y==3, else 0)
      - ChoseMajority: binary (1 if y==2 else 0)
      - age_c: centered age (age - mean(age))
      - AgeGroup: coarse categorical age bins for descriptive checks
      - Culture: categorical version of culture
      - IsMale: binary gender indicator (1 = boy)
      - MajorityFirst: copy of majority_first as int
      - School: categorical school identifier (used for clustering)

    Keeps relevant control columns (religiousness, calworks) as numeric.
    """
    df = df.copy()

    # Drop rows with missing key variables
    df = df.dropna(subset=['y', 'age', 'culture'])

    # Binary outcomes
    # y: 1 = unchosen (undemonstrated), 2 = majority, 3 = minority
    df['ChoseDemonstrated'] = df['y'].apply(lambda v: 1 if v in [2, 3] else 0).astype(int)
    df['ChoseMajority'] = df['y'].apply(lambda v: 1 if v == 2 else 0).astype(int)

    # Center age for interpretability in interactions
    df['age_c'] = df['age'] - df['age'].mean()

    # Coarse developmental bins for description (kept as a column for plots / checks)
    # bins: 4-6, 7-9, 10-14
    df['AgeGroup'] = pd.cut(df['age'], bins=[3, 6, 9, 14], labels=['4-6', '7-9', '10-14'], right=True)

    # Culture as categorical factor
    df['Culture'] = df['culture'].astype('category')

    # Gender mapping: dataset 1 = girl, 2 = boy
    df['IsMale'] = df['gender'].map({1: 0, 2: 1}).astype('Int64').fillna(0).astype(int)

    # Order / presentation control
    df['MajorityFirst'] = df['majority_first'].fillna(0).astype(int)

    # Ensure controls are numeric (preserve NaNs if present)
    if 'religiousness' in df.columns:
        df['religiousness'] = pd.to_numeric(df['religiousness'], errors='coerce')
    else:
        df['religiousness'] = np.nan

    if 'calworks' in df.columns:
        df['calworks'] = pd.to_numeric(df['calworks'], errors='coerce')
    else:
        df['calworks'] = np.nan

    # School as categorical for clustering
    if 'school' in df.columns:
        # Fill missing school with a placeholder to allow clustering; cluster code will still work
        df['School'] = df['school'].fillna('Unknown').astype('category')
    else:
        df['School'] = 'Unknown'

    # Return only the columns needed for analysis plus a few descriptive columns
    keep_cols = [
        'y', 'ChoseDemonstrated', 'ChoseMajority', 'age', 'age_c', 'AgeGroup', 'Culture',
        'IsMale', 'MajorityFirst', 'religiousness', 'calworks', 'School'
    ]
    # Add any of these if present in df
    keep_cols = [c for c in keep_cols if c in df.columns]

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit two primary models to answer the research question:
      1) Whether children rely on social information at all (ChoseDemonstrated) as a function of age, culture, and their interaction, controlling for gender, order, religiousness and calworks.
      2) Among children who chose a demonstrated option, whether they prefer the majority (ChoseMajority) with the same predictors.

    Implementation details:
      - Both models are binomial GLMs (logistic regression) fitted with statsmodels.
      - Standard errors are cluster-robust at the School level to account for non-independence within schools.

    Returns a dict with fitted (robust) result objects for both models.
    """
    import statsmodels.formula.api as smf

    results = {}

    # Model formula: main effects + interaction between age and culture
    formula = 'ChoseDemonstrated ~ age_c * C(Culture) + IsMale + MajorityFirst + religiousness + calworks'

    # Fit GLM (binomial) for reliance on social information
    glm1 = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
    res1 = glm1.fit()

    # Cluster-robust SEs by School (if School exists)
    try:
        res1_clust = res1.get_robustcov_results(cov_type='cluster', groups=df['School'])
    except Exception:
        # Fall back to default if clustering fails
        res1_clust = res1

    results['ChoseDemonstrated_model'] = res1_clust

    # Model 2: preference for majority among those who used social information
    df_dem = df[df['ChoseDemonstrated'] == 1].copy()

    # Require a minimal sample size to fit sensible model
    if df_dem.shape[0] < 30:
        results['ChoseMajority_model'] = None
    else:
        formula2 = 'ChoseMajority ~ age_c * C(Culture) + IsMale + MajorityFirst + religiousness + calworks'
        glm2 = smf.glm(formula=formula2, data=df_dem, family=sm.families.Binomial())
        res2 = glm2.fit()
        try:
            res2_clust = res2.get_robustcov_results(cov_type='cluster', groups=df_dem['School'])
        except Exception:
            res2_clust = res2
        results['ChoseMajority_model'] = res2_clust

    # Optionally print summaries for quick inspection (comment out if used programmatically)
    try:
        print('\n--- ChoseDemonstrated model (cluster-robust results) ---')
        print(results['ChoseDemonstrated_model'].summary())
        if results['ChoseMajority_model'] is not None:
            print('\n--- ChoseMajority model (subset; cluster-robust results) ---')
            print(results['ChoseMajority_model'].summary())
        else:
            print('\nChoseMajority model not fit: too few demonstrated-choice cases (n < 30).')
    except Exception:
        # If printing fails (e.g., when called in non-interactive context), ignore
        pass

    return results


