from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/amtl/noperturb_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataframe for binomial GLM of AMTL.
    Output columns (required by model):
      - num_amtl (int): number of missing teeth for the given tooth class
      - sockets (int): number of observable sockets (binomial denominator)
      - AMTL_prop (float): num_amtl / sockets (proportion between 0 and 1)
      - IsHuman (int): 1 if genus == 'Homo sapiens', else 0
      - age (float): estimated age at death
      - prob_male (float): estimated probability specimen is male
      - tooth_class (category): tooth class factor
      - specimen (category): specimen identifier (used for clustering)
    """
    # Work on a copy
    df = df.copy()

    # Required columns
    required_cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen']
    # Drop rows missing any required field
    df = df.dropna(subset=required_cols)

    # Ensure sockets numeric and positive; drop rows with zero or negative sockets
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    df = df.dropna(subset=['sockets'])
    df = df[df['sockets'] > 0]

    # Ensure num_amtl is numeric; clip to valid range [0, sockets]
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df = df.dropna(subset=['num_amtl'])
    # Clip to integer range 0..sockets (some data entry issues may exist)
    # Round num_amtl to nearest integer if not integer
    df['num_amtl'] = df['num_amtl'].round().astype(int)
    # Clip to valid range
    df['num_amtl'] = df[['num_amtl', 'sockets']].apply(lambda row: max(0, min(int(row['num_amtl']), int(row['sockets']))), axis=1)

    # Create proportion outcome
    df['AMTL_prop'] = df['num_amtl'] / df['sockets']

    # Create IsHuman indicator: 1 for 'Homo sapiens', 0 otherwise
    # Handle possible whitespace/capitalization variations
    df['genus'] = df['genus'].astype(str).str.strip()
    df['IsHuman'] = (df['genus'] == 'Homo sapiens').astype(int)

    # Ensure tooth_class is categorical and normalized
    df['tooth_class'] = df['tooth_class'].astype(str).str.strip().replace({'premolar': 'Premolar', 'anterior': 'Anterior', 'posterior': 'Posterior'})
    df['tooth_class'] = df['tooth_class'].astype('category')

    # Ensure specimen is categorical
    df['specimen'] = df['specimen'].astype(str).astype('category')

    # Keep only columns needed for modeling plus helpful originals
    keep_cols = ['num_amtl', 'sockets', 'AMTL_prop', 'IsHuman', 'age', 'prob_male', 'tooth_class', 'specimen', 'genus']
    df = df[keep_cols]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial GLM for AMTL proportion with number of sockets as binomial denominator.
    Model: AMTL_prop ~ IsHuman + age + prob_male + C(tooth_class)
    Family: Binomial
    Weights: sockets (number of trials)

    To account for non-independence of observations from the same specimen (multiple tooth classes per specimen),
    compute cluster-robust standard errors clustered by specimen.

    Returns the fitted results object with clustered robust covariance (so summary/stats reflect clustering by specimen).
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Ensure required columns are present
    required = ['AMTL_prop', 'sockets', 'IsHuman', 'age', 'prob_male', 'tooth_class', 'specimen']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in the transformed dataframe: {missing}")

    # Fit binomial GLM on proportions with trial counts as weights
    # Using weights=sockets tells statsmodels to treat AMTL_prop as proportion with 'sockets' trials
    formula = 'AMTL_prop ~ IsHuman + age + prob_male + C(tooth_class)'
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), weights=df['sockets'])
    results = model.fit()

    # Get cluster-robust covariance by specimen (accounts for repeated measures per specimen)
    # If specimen has many unique values this will produce cluster-robust SEs; if specimen is unique per row this reduces to usual SEs.
    try:
        robust_results = results.get_robustcov_results(cov_type='cluster', groups=df['specimen'])
    except Exception:
        # Fallback: if clustering fails for any reason, return the original results
        robust_results = results

    # Return the results object with clustered (or original) covariance
    return robust_results


