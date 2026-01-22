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
    Prepare dataframe for binomial GLM of AMTL.

    Transformations performed:
    - Drop rows with missing critical fields (num_amtl, sockets, genus, tooth_class, age, prob_male).
    - Remove rows with non-positive sockets.
    - Ensure integer counts and cap num_amtl at sockets if necessary.
    - Sanitize genus values (replace spaces with underscore so factor names are safe for dummies).
    - Create proportion column 'prop_amtl' = num_amtl / sockets (for inspection only).
    - Standardize (z-score) age into 'age_z' to aid model convergence/interpretation.
    - Create indicator 'is_human' (1 if genus == 'Homo_sapiens', 0 otherwise) for convenience / post-hoc checks.

    Returns transformed dataframe containing at least these columns used in modeling:
    ['num_amtl', 'sockets', 'genus', 'tooth_class', 'age_z', 'prob_male', 'prop_amtl', 'is_human']
    """
    df = df.copy()

    # Required columns for analysis
    required_cols = ['num_amtl', 'sockets', 'genus', 'tooth_class', 'age', 'prob_male']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    # Drop rows missing required data
    df = df.dropna(subset=required_cols)

    # Keep only rows with positive (non-zero) sockets
    df = df[df['sockets'] > 0]

    # Ensure integer counts and sensible ranges
    df['num_amtl'] = df['num_amtl'].astype(int)
    df['sockets'] = df['sockets'].astype(int)

    # If num_amtl > sockets (data error), cap to sockets
    df['num_amtl'] = df[['num_amtl', 'sockets']].min(axis=1)

    # Sanitize genus labels so they are safe as column names (e.g., 'Homo sapiens' -> 'Homo_sapiens')
    df['genus'] = df['genus'].astype(str).str.strip().str.replace('\n', ' ').str.replace(' ', '_')

    # Ensure tooth_class is a string categorical
    df['tooth_class'] = df['tooth_class'].astype(str).str.strip()

    # Proportion (useful for EDA). Kept for completeness but the model uses counts.
    df['prop_amtl'] = df['num_amtl'] / df['sockets']

    # Standardize age (z-score) for model stability
    age_mean = df['age'].mean()
    age_std = df['age'].std(ddof=0)
    if age_std == 0 or np.isnan(age_std):
        df['age_z'] = 0.0
    else:
        df['age_z'] = (df['age'] - age_mean) / age_std

    # Convenience indicator for human specimens
    df['is_human'] = (df['genus'] == 'Homo_sapiens').astype(int)

    # Return only rows/columns necessary for modeling & inspection
    # (we keep full df but guarantee these columns exist)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit a binomial GLM predicting probability of AMTL (missing tooth) per socket.

    Modeling approach:
    - Use endog as a two-column array of successes/failures: [num_amtl, sockets - num_amtl].
    - Build design matrix with categorical dummies for 'genus' and 'tooth_class' (drop_first=True),
      plus continuous controls 'age_z' and 'prob_male'. Intercept is added explicitly.
    - Fit a GLM with Binomial family using statsmodels.api.GLM and return the fitted results object.

    Returns:
    - statsmodels GLMResults object (call .summary() on it to inspect coefficients and p-values).
    """
    import statsmodels.api as sm

    df = df.copy()

    # Basic validation
    for c in ['num_amtl', 'sockets', 'genus', 'tooth_class', 'age_z', 'prob_male']:
        if c not in df.columns:
            raise ValueError(f"Transformed dataframe must contain column '{c}' for modeling")

    # Build design matrix: categorical dummies for genus and tooth_class
    cat_df = pd.get_dummies(df[['genus', 'tooth_class']], drop_first=True)

    # Add continuous controls
    cont_df = df[['age_z', 'prob_male', 'is_human']].copy()

    X = pd.concat([cat_df, cont_df], axis=1)
    X = sm.add_constant(X, has_constant='add')

    # Endogenous: two-column (successes, failures) array required by statsmodels for binomial counts
    successes = df['num_amtl'].astype(int).values
    failures = (df['sockets'] - df['num_amtl']).astype(int).values
    endog = np.vstack([successes, failures]).T

    # Fit GLM with Binomial family
    model = sm.GLM(endog, X, family=sm.families.Binomial())
    results = model.fit()

    # Return the fitted results object for downstream inspection
    return results


