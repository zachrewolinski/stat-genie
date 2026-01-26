from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/teachingratings/replace_with_rvs_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Hamermesh & Parker (2005) course-evaluation dataset into a dataframe
    suitable for OLS modeling of the effect of instructor beauty on student evaluations.

    Steps performed:
    - Make a defensive copy of the dataframe.
    - Drop observations missing the key outcome ('eval') or the key predictor ('beauty').
    - Ensure categorical columns are cast to category dtype.
    - Create a mean-centered beauty variable 'beauty_c' (used in the model).
    - Create a log-transformed class-size variable 'log_students' from 'students'.
    - Coerce 'age' to numeric and drop rows missing any column required for the model.

    Returns the transformed dataframe containing the columns referenced in the modeling code.
    """
    df = df.copy()

    # Required columns for modeling
    required_cols = ['beauty', 'eval']
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"Required column '{c}' not in input dataframe")

    # Drop rows missing the main variables
    df = df.dropna(subset=['beauty', 'eval'])

    # Ensure categorical columns exist before casting
    categorical_cols = ['gender', 'minority', 'credits', 'division', 'native', 'tenure', 'prof']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # Center beauty for interpretability of main effect and interactions
    df['beauty_c'] = df['beauty'] - df['beauty'].mean()

    # Coerce age to numeric if present
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Create log-transformed students variable to capture class size (log1p to be safe)
    if 'students' in df.columns:
        # coerce to numeric then log transform; keep zero-safe
        df['students'] = pd.to_numeric(df['students'], errors='coerce')
        df['log_students'] = np.log1p(df['students'])
    else:
        # If 'students' not present create a placeholder with NaNs
        df['log_students'] = np.nan

    # Final set of columns the model uses; drop rows missing any of these
    model_cols = ['eval', 'beauty_c', 'gender', 'age', 'minority', 'tenure', 'native', 'division', 'credits', 'log_students', 'prof']
    # Keep only columns that actually exist in df when deciding to dropna
    model_cols_in_df = [c for c in model_cols if c in df.columns]

    df = df.dropna(subset=model_cols_in_df)

    # Return transformed dataframe (contains the exact column names used in the model)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit an OLS regression estimating the effect of instructor beauty on student evaluations.

    Model specification:
    - Dependent variable: eval
    - Independent variable: beauty_c (mean-centered beauty)
    - Moderator: gender (interaction beauty_c * gender)
    - Controls: age, minority, tenure, native, division, credits, log_students
    - Professor fixed effects: C(prof)
    - Robust standard errors clustered by professor id

    Returns a dict with the raw OLS fit and the clustered-robust covariance fit.
    """
    import statsmodels.formula.api as smf

    # Build formula. We explicitly use categorical encoding via C() for factors.
    formula = (
        'eval ~ beauty_c * C(gender) + age + C(minority) + C(tenure) + '
        'C(native) + C(division) + C(credits) + log_students + C(prof)'
    )

    # Fit the baseline OLS model (with dummy fixed effects included via C(prof))
    ols_result = smf.ols(formula, data=df).fit()

    # Obtain clustered (by prof) robust covariance results
    # Use get_robustcov_results to apply cluster-robust standard errors
    clustered_result = ols_result.get_robustcov_results(cov_type='cluster', groups=df['prof'])

    # Print summaries for quick inspection (user can inspect returned objects programmatically)
    try:
        print('\n--- OLS summary (conventional SEs) ---')
        print(ols_result.summary())
    except Exception:
        pass
    try:
        print('\n--- OLS summary (cluster-robust SEs clustered by prof) ---')
        print(clustered_result.summary())
    except Exception:
        pass

    # Return both results so downstream code can inspect coefficients and robust SEs
    return {'ols': ols_result, 'clustered': clustered_result}


