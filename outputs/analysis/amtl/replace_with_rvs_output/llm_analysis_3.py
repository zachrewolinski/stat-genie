from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/replace_with_rvs_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataset for binomial GLM of AMTL.

    Transformations performed:
    - Drop rows with missing or invalid essential values (num_amtl, sockets, age, prob_male, tooth_class, genus).
    - Ensure sockets > 0 (rows with sockets <= 0 are removed).
    - Cap num_amtl to sockets if num_amtl > sockets (data cleaning guard).
    - Create proportion column prop_amtl = num_amtl / sockets.
    - Create binary indicator IsHomo = 1 if genus == 'Homo sapiens', else 0.
    - Ensure tooth_class is categorical and standardize categories (strip whitespace).

    Returns a dataframe containing at minimum the columns used in modeling:
    ['num_amtl', 'sockets', 'prop_amtl', 'IsHomo', 'age', 'prob_male', 'tooth_class', 'genus', 'specimen']
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Required columns list
    required_cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus']

    # Drop rows with missing values in required columns
    df = df.dropna(subset=required_cols)

    # Ensure sockets is numeric
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce')

    # Drop rows with missing/invalid sockets or num_amtl after coercion
    df = df.dropna(subset=['sockets', 'num_amtl'])

    # Keep only rows with sockets > 0
    df = df[df['sockets'] > 0]

    # Cap num_amtl to sockets (data cleaning) and ensure non-negative
    df['num_amtl'] = df['num_amtl'].clip(lower=0)
    df['num_amtl'] = df[['num_amtl', 'sockets']].apply(lambda row: min(row['num_amtl'], row['sockets']), axis=1)

    # Create proportion outcome
    df['prop_amtl'] = df['num_amtl'] / df['sockets']

    # Binary indicator for Homo sapiens
    df['IsHomo'] = df['genus'].astype(str).str.strip().apply(lambda x: 1 if x == 'Homo sapiens' else 0)

    # Clean tooth_class and make categorical
    df['tooth_class'] = df['tooth_class'].astype(str).str.strip()
    df['tooth_class'] = pd.Categorical(df['tooth_class'], categories=['Anterior', 'Premolar', 'Posterior'])

    # Keep commonly used columns and return
    keep_cols = ['specimen', 'genus', 'tooth_class', 'num_amtl', 'sockets', 'prop_amtl', 'IsHomo', 'age', 'prob_male', 'stdev_age']
    for c in keep_cols:
        if c not in df.columns:
            # If optional columns are missing, create as NA to keep consistent columns
            df[c] = pd.NA

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a binomial (logistic) GLM for AMTL using counts as binomial trials.

    Model specification:
    - Response: proportion prop_amtl with number of trials = sockets (implemented via weights).
    - Predictors: IsHomo (test of interest), age (continuous), prob_male (continuous), and tooth_class (categorical).

    The function returns the fitted GLM results object.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Ensure required columns are present
    required = ['prop_amtl', 'sockets', 'IsHomo', 'age', 'prob_male', 'tooth_class']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Drop rows with missing predictor or response values
    model_df = df.dropna(subset=required).copy()

    # Build formula. Use C(tooth_class) to include categorical tooth class dummies.
    formula = 'prop_amtl ~ IsHomo + age + prob_male + C(tooth_class)'

    # Fit GLM with Binomial family. Use 'sockets' as the frequency (var_weights) to indicate number of trials.
    # In statsmodels, when modeling proportions, pass the proportion as endog and use var_weights=number_of_trials.
    glm_binom = smf.glm(formula=formula, data=model_df, family=sm.families.Binomial(), var_weights=model_df['sockets'])
    results = glm_binom.fit()

    # Return the fitted results object. The caller can examine results.summary() or results.params etc.
    return results


