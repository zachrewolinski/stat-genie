from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/projects/binyu/hao_huang/stat-genie/.venv/lib/python3.11/site-packages/blade_bench/datasets/hurricane/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the hurricane dataset for modeling.
    - Drop rows missing any variables required for the model.
    - Ensure numeric types for modeling columns.
    - Create a standardized femininity score (masfem_z) for interpretability.
    - Ensure category and gender_mf are integer codes.
    Returns a dataframe containing at minimum the columns referenced in the conceptual model:
    ['alldeaths', 'masfem_z', 'wind', 'min', 'category', 'year', 'gender_mf']
    """
    df = df.copy()

    # Columns we require for the model
    required_cols = ['alldeaths', 'masfem', 'wind', 'min', 'category', 'year', 'gender_mf']

    # Force numeric types where appropriate and coerce parsing errors to NaN
    for c in required_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing any required columns
    df = df.dropna(subset=required_cols)

    # Standardize the masfem score (z-score) for interpretability
    # Use population std (ddof=0) to avoid small-sample adjustment; either is acceptable
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / df['masfem'].std(ddof=0)

    # Ensure integer coding for category and gender_mf
    # category is ordinal (1-5); cast to int
    df['category'] = df['category'].astype(int)
    # gender_mf should be binary 0/1
    df['gender_mf'] = df['gender_mf'].astype(int)

    # Keep only columns necessary for downstream modeling + original masfem for reference
    keep_cols = ['alldeaths', 'masfem', 'masfem_z', 'wind', 'min', 'category', 'year', 'gender_mf']
    existing_keep = [c for c in keep_cols if c in df.columns]
    df = df[existing_keep].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a negative binomial regression predicting hurricane fatalities (alldeaths)
    from standardized femininity of the hurricane name (masfem_z), controlling for
    storm intensity and temporal trends. Category is dummy-coded (drop-first).

    Returns the fitted results object from statsmodels.
    """
    import statsmodels.api as sm
    import pandas as pd

    # Ensure required columns exist
    required = ['alldeaths', 'masfem_z', 'wind', 'min', 'category', 'year', 'gender_mf']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in transformed dataframe: {missing}")

    # Outcome
    y = df['alldeaths'].astype(float)

    # Base covariates
    X_base = df[['masfem_z', 'wind', 'min', 'year', 'gender_mf']].copy()

    # Dummy-code the ordinal category to allow flexible control (drop first level to avoid multicollinearity)
    cat_dummies = pd.get_dummies(df['category'].astype(str), prefix='cat', drop_first=True)

    # Combine regressors
    X = pd.concat([X_base, cat_dummies], axis=1)

    # Add intercept
    X = sm.add_constant(X, has_constant='add')

    # Fit Negative Binomial model to account for overdispersion common in count data
    # Use GLM with NegativeBinomial family
    model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
    results = model.fit()

    # Print a brief summary for immediate inspection (caller can inspect returned results object)
    print(results.summary())

    return results


