from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/teachingratings/replace_with_rvs_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the Hamermesh classroom dataset for modeling the effect of beauty on evaluations.

    Transformations performed:
    - Copy input dataframe to avoid side effects.
    - Drop rows with missing values in variables required for the analysis.
    - Ensure categorical variables have category dtype so formula interface treats them as factors (C()).
    - Create a squared beauty term (beauty_sq) to model nonlinearity.

    The returned dataframe contains at minimum the columns named in the conceptual variables:
    'eval', 'beauty', 'beauty_sq', 'age', 'students', 'allstudents', 'gender', 'minority',
    'tenure', 'division', 'credits', 'native', 'prof'

    """
    # Work on a copy
    df = df.copy()

    # Required columns for analysis
    required_cols = [
        'eval', 'beauty', 'age', 'students', 'allstudents',
        'gender', 'minority', 'tenure', 'division', 'credits', 'native', 'prof'
    ]

    # Drop rows missing any required column
    df = df.dropna(subset=required_cols)

    # Ensure categorical columns are categorical (helps formula interface and clarity)
    cat_cols = ['gender', 'minority', 'tenure', 'division', 'credits', 'native']
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype('category')

    # Ensure 'prof' is categorical-like for clustering (but keep numeric id for clustering API)
    # If prof is not numeric, attempt to convert; otherwise, leave as-is
    if 'prof' in df.columns:
        try:
            df['prof'] = pd.to_numeric(df['prof'], errors='raise')
        except Exception:
            # If conversion fails, create a numeric code
            df['prof'] = df['prof'].astype('category').cat.codes

    # Create squared beauty term to capture nonlinearity
    df['beauty_sq'] = df['beauty'] ** 2

    # Return the transformed df used by the model
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Estimate the impact of beauty on teaching evaluations using OLS with controls
    and cluster-robust standard errors at the instructor (prof) level.

    Model specification:
    eval ~ beauty + beauty_sq + age + students + allstudents + C(gender) + C(minority)
           + C(tenure) + C(division) + C(credits) + C(native)

    Returns the fitted statsmodels results object (with clustered robust cov).
    """
    import statsmodels.formula.api as smf

    # Check that required columns exist
    required = [
        'eval', 'beauty', 'beauty_sq', 'age', 'students', 'allstudents',
        'gender', 'minority', 'tenure', 'division', 'credits', 'native', 'prof'
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataframe: {missing}")

    # Define formula with categorical controls wrapped in C()
    formula = (
        'eval ~ beauty + beauty_sq + age + students + allstudents '
        '+ C(gender) + C(minority) + C(tenure) + C(division) + C(credits) + C(native)'
    )

    # Fit OLS
    mod = smf.ols(formula, data=df)

    # Fit and request cluster-robust covariance by professor id
    # If 'prof' has many unique values, clustering is appropriate for instructor-level correlation.
    results = mod.fit(cov_type='cluster', cov_kwds={'groups': df['prof']})

    # Return the fitted results object (caller can call .summary() or inspect params)
    return results


