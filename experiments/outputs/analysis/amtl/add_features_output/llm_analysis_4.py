from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/add_features_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset to the analysis-ready dataframe. The returned dataframe will contain these columns used in the model:
      - num_amtl (int): number of missing teeth in the tooth class
      - sockets (int): number of observable sockets for that tooth class (binomial denominator)
      - amtl_rate (float): num_amtl / sockets
      - genus (category): cleaned genus name (e.g., 'Homo sapiens', 'Pan', 'Pongo', 'Papio')
      - age (float): estimated age at death
      - prob_male (float): estimated probability specimen is male (0-1)
      - tooth_class (category): tooth class (Anterior, Posterior, Premolar)
      - specimen (object): specimen identifier (kept for clustering)

    Steps:
      - Drop rows missing essential fields (num_amtl, sockets, genus, age, prob_male, tooth_class, specimen).
      - Ensure numeric types, clamp num_amtl between 0 and sockets.
      - Create amtl_rate = num_amtl / sockets.
      - Convert genus and tooth_class to categorical with consistent level names.
    """

    # Work on a copy
    df = df.copy()

    # Required columns
    required_cols = ['num_amtl', 'sockets', 'genus', 'age', 'prob_male', 'tooth_class', 'specimen']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns in input dataframe: {missing}")

    # Drop rows with missing essential data
    df = df.dropna(subset=required_cols)

    # Ensure integer numeric types for counts
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce').round().astype(int)
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce').round().astype(int)

    # Drop rows with non-positive sockets
    df = df[df['sockets'] > 0].copy()

    # Make sure num_amtl is in [0, sockets]
    df.loc[df['num_amtl'] < 0, 'num_amtl'] = 0
    df.loc[df['num_amtl'] > df['sockets'], 'num_amtl'] = df.loc[df['num_amtl'] > df['sockets'], 'sockets']

    # Numeric controls
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')

    # Drop any rows that became NaN after coercion
    df = df.dropna(subset=['age', 'prob_male'])

    # Create proportion outcome (binomial) --- will be used with sockets as weights
    df['amtl_rate'] = df['num_amtl'] / df['sockets']

    # Clean and standardize genus strings
    df['genus'] = df['genus'].astype(str).str.strip()
    # Map common variants to canonical names (best-effort)
    df['genus'] = df['genus'].replace({
        'Homo': 'Homo sapiens',
        'Homo sapiens sapiens': 'Homo sapiens',
        'H. sapiens': 'Homo sapiens'
    })
    # Keep only the genera of interest; drop anything else (if present)
    valid_genera = ['Homo sapiens', 'Pan', 'Pongo', 'Papio']
    df = df[df['genus'].isin(valid_genera)].copy()

    # Standardize tooth_class names and make categorical
    df['tooth_class'] = df['tooth_class'].astype(str).str.strip().str.title()
    # Accept common synonyms and map them to canonical three classes
    df['tooth_class'] = df['tooth_class'].replace({
        'Anterior': 'Anterior',
        'Posterior': 'Posterior',
        'Posteriors': 'Posterior',
        'Premolar': 'Premolar',
        'Premolars': 'Premolar',
        'Molar': 'Posterior'
    })
    df = df[df['tooth_class'].isin(['Anterior', 'Posterior', 'Premolar'])].copy()

    # Convert categorical columns to category dtype
    df['genus'] = pd.Categorical(df['genus'], categories=['Homo sapiens', 'Pan', 'Pongo', 'Papio'])
    df['tooth_class'] = pd.Categorical(df['tooth_class'], categories=['Anterior', 'Premolar', 'Posterior'])

    # Keep only final columns needed for modeling (plus specimen for clustering)
    final_cols = ['num_amtl', 'sockets', 'amtl_rate', 'genus', 'age', 'prob_male', 'tooth_class', 'specimen']
    df = df[final_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a binomial (logistic) regression model for AMTL using a GLM with binomial family.

    Model specification:
      amtl_rate ~ C(genus) + age + prob_male + C(tooth_class)
    where amtl_rate = num_amtl / sockets, and sockets are used as binomial weights.

    To account for non-independence of multiple observations from the same specimen (rows per tooth_class),
    the model returns cluster-robust standard errors clustered by 'specimen'.

    Returns the fitted GLMResults object (statsmodels) so the user can inspect coefficients, CIs, and summaries.
    """

    # Ensure required columns exist
    required = ['num_amtl', 'sockets', 'amtl_rate', 'genus', 'age', 'prob_male', 'tooth_class', 'specimen']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Build formula using the proportion as the endogenous variable; we'll pass sockets as var_weights
    formula = 'amtl_rate ~ C(genus) + age + prob_male + C(tooth_class)'

    # Fit the binomial GLM with var_weights = sockets (tells GLM the number of trials per observation)
    # Use clustered standard errors by specimen to account for within-specimen correlation.
    model = sm.GLM.from_formula(formula,
                                data=df,
                                family=sm.families.Binomial(),
                                var_weights=df['sockets'])

    # Fit the model; compute cluster-robust SEs clustered on specimen
    results = model.fit()

    # Recompute robust covariance clustered by specimen (if specimen has >1 observation per cluster, this will adjust SEs)
    try:
        clustered_results = results.get_robustcov_results(cov_type='cluster', groups=df['specimen'])
        # Return clustered_results which has cluster-robust SEs
        return clustered_results
    except Exception:
        # If clustered robust cov could not be computed (rare), return the original results
        return results

# Example of usage (outside of this function):
# df2 = transform(raw_df)
# res = model(df2)
# print(res.summary())


