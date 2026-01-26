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
    Prepares the dataset for modeling. Key steps:
    - Keep only rows with non-missing values for variables used in the model.
    - Coerce categorical columns to pandas.Categorical dtype.
    - Standardize the beauty variable to a z-score (beauty_z).
    - Create a log transform of students (students_log).
    - Return a dataframe containing only the columns used in the statistical model.
    """
    df = df.copy()

    # Required raw columns
    required_cols = ['beauty', 'eval', 'age', 'gender', 'minority', 'tenure', 'native', 'division', 'credits', 'students', 'prof']

    # Drop rows with missing values in required columns
    df = df.dropna(subset=required_cols)

    # Ensure correct dtypes for categorical variables
    for c in ['gender', 'minority', 'tenure', 'native', 'division', 'credits', 'prof']:
        # If any of these are numeric codes, treat them as strings first
        df[c] = df[c].astype('category')

    # Standardize the beauty rating (z-score). Use population std (ddof=0) for interpretability.
    df['beauty_z'] = (df['beauty'] - df['beauty'].mean()) / (df['beauty'].std(ddof=0) if df['beauty'].std(ddof=0) != 0 else 1.0)

    # Log-transform the number of students who participated to reduce skew
    # Add a small constant guard in case of zeros (shouldn't occur given schema min=5)
    df['students_log'] = np.log(df['students'].astype(float) + 1e-6)

    # Keep only the columns needed for modeling
    out_cols = ['beauty_z', 'eval', 'age', 'gender', 'minority', 'tenure', 'native', 'division', 'credits', 'students_log', 'prof']
    df = df[out_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fits two complementary models to estimate the association between instructor beauty and student evaluations:
    1) Mixed-effects linear model with a random intercept for professor (accounts for multiple courses per instructor).
    2) OLS with standard errors clustered by professor (robustness check).

    Returns a dictionary with the fitted model results objects.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Ensure incoming df contains required columns
    required = ['beauty_z', 'eval', 'age', 'gender', 'minority', 'tenure', 'native', 'division', 'credits', 'students_log', 'prof']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Transformed dataframe is missing required columns: {missing}")

    # Formula: main effect of beauty plus controls; categorical controls are wrapped with C(...)
    formula = (
        "eval ~ beauty_z + age + C(gender) + C(minority) + C(tenure) + C(native) + "
        "C(division) + C(credits) + students_log"
    )

    results = {}

    # 1) Mixed effects model with random intercept for professor
    try:
        md = smf.mixedlm(formula, df, groups=df['prof'])
        mdf = md.fit(reml=False)
        results['mixedlm'] = mdf
    except Exception as e:
        results['mixedlm_error'] = str(e)

    # 2) OLS with clustered standard errors by professor (robustness)
    try:
        ols = smf.ols(formula, df).fit()
        # clustered SEs by 'prof'
        ols_cluster = ols.get_robustcov_results(cov_type='cluster', groups=df['prof'])
        results['ols_clustered'] = ols_cluster
    except Exception as e:
        results['ols_error'] = str(e)

    return results


