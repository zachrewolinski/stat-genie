from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/noperturb_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe ready for binomial GLM.

    Adds/returns the following columns (used in modeling):
      - n_missing: integer number of teeth missing for the row (num_amtl)
      - trials: integer number of observable sockets (sockets)
      - prop_missing: n_missing / trials (for inspection)
      - is_homo: 1 if genus == 'Homo sapiens', otherwise 0
      - age_c: centered age (age - mean(age))
      - prob_male: carried through (should be between 0 and 1)
      - tooth_Premolar, tooth_Posterior: dummy columns (Anterior reference)
      - specimen: kept as-is (for clustering)

    Cleaning rules:
      - Drop rows with missing values in key columns: num_amtl, sockets, genus, tooth_class, age, prob_male, specimen
      - Drop rows with non-positive sockets or where num_amtl > sockets
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Required columns
    required_cols = ['num_amtl', 'sockets', 'genus', 'tooth_class', 'age', 'prob_male', 'specimen']
    missing_req = [c for c in required_cols if c not in df.columns]
    if missing_req:
        raise KeyError(f"Missing required columns in input df: {missing_req}")

    # Drop rows with NA in required fields
    df = df.dropna(subset=required_cols)

    # Ensure numeric types where applicable
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')

    # Drop rows that became NA after coercion
    df = df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male'])

    # Remove rows with invalid trial counts
    df = df[df['sockets'] > 0]

    # Remove rows where num_amtl is out of plausible range relative to sockets
    df = df[(df['num_amtl'] >= 0) & (df['num_amtl'] <= df['sockets'])]

    # Create modeling columns
    df['n_missing'] = df['num_amtl'].astype(int)
    df['trials'] = df['sockets'].astype(int)
    df['prop_missing'] = df['n_missing'] / df['trials']

    # Independent variable: is_homo indicator
    df['is_homo'] = (df['genus'].astype(str).str.strip() == 'Homo sapiens').astype(int)

    # Center age to improve numeric stability
    age_mean = df['age'].mean()
    df['age_c'] = df['age'] - age_mean

    # Ensure prob_male is bounded [0,1]
    df['prob_male'] = df['prob_male'].clip(0.0, 1.0)

    # Create tooth_class dummies; use Anterior as reference by drop_first=True
    tooth_dummies = pd.get_dummies(df['tooth_class'].astype(str).str.strip(), prefix='tooth', drop_first=True)
    # Expected columns: tooth_Premolar, tooth_Posterior (if those categories present)
    for col in tooth_dummies.columns:
        df[col] = tooth_dummies[col]

    # If a dummy column is missing because category not present, add it with zeros
    for expected in ['tooth_Premolar', 'tooth_Posterior']:
        if expected not in df.columns:
            df[expected] = 0

    # Keep specimen as string identifier for clustering
    df['specimen'] = df['specimen'].astype(str)

    # Final check: drop any rows where trials <= 0 or n_missing is NA
    df = df.dropna(subset=['n_missing', 'trials'])
    df = df[df['trials'] > 0]

    # Return only columns needed for downstream analysis plus a few for inspection
    cols_keep = [
        'specimen', 'genus', 'tooth_class', 'n_missing', 'trials', 'prop_missing',
        'is_homo', 'age_c', 'prob_male', 'tooth_Premolar', 'tooth_Posterior'
    ]
    # Add any columns that exist but were not in the list (safe-guard)
    cols_present = [c for c in cols_keep if c in df.columns]
    return df[cols_present]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial GLM predicting the count of antemortem tooth loss (n_missing) out of trials (sockets)
    with primary predictor is_homo and controls age_c, prob_male, and tooth class dummies.

    Uses clustered (by specimen) robust standard errors to account for non-independence of observations
    coming from the same specimen.

    Returns the fitted GLMResults object.
    """
    # Ensure required transformed columns are present
    required = ['n_missing', 'trials', 'is_homo', 'age_c', 'prob_male', 'tooth_Premolar', 'tooth_Posterior', 'specimen']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for modeling: {missing}")

    # Build endogenous variable as a two-column array [[successes, failures], ...]
    endog = np.vstack([df['n_missing'].values, (df['trials'] - df['n_missing']).values]).T

    # Build exogenous matrix (design matrix)
    exog_cols = ['is_homo', 'age_c', 'prob_male', 'tooth_Premolar', 'tooth_Posterior']
    exog = df[exog_cols].astype(float)
    exog = sm.add_constant(exog, has_constant='add')

    # Fit binomial GLM
    model_glm = sm.GLM(endog, exog, family=sm.families.Binomial())

    # Fit and obtain clustered robust standard errors by specimen
    try:
        results = model_glm.fit(cov_type='cluster', cov_kwds={'groups': df['specimen']})
    except Exception:
        # Fallback to default fit if clustering fails for some reason
        results = model_glm.fit()

    # Attach some metadata for interpretation
    results.model_data = {
        'exog_cols': exog_cols,
        'n_obs': len(df),
        'n_specimens': df['specimen'].nunique()
    }

    return results


