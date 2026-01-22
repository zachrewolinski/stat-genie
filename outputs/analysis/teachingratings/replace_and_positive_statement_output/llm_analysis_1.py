from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/teachingratings/replace_and_positive_statement_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Hamermesh classroom dataset into a modeling dataframe.

    Produces the following columns used in the model:
      - eval: dependent variable (numeric)
      - beauty_z: standardized beauty score (mean 0, sd 1)
      - age: instructor age (numeric)
      - gender_male: 1 if gender == 'male', 0 if 'female'
      - minority_yes: 1 if minority == 'yes', 0 if 'no'
      - credits_single: 1 if credits == 'single', 0 if 'more'
      - division_lower: 1 if division == 'lower', 0 if 'upper'
      - native_yes: 1 if native == 'yes', 0 if 'no'
      - tenure_yes: 1 if tenure == 'yes', 0 if 'no'
      - log_students: natural log of students (participating in evaluation)
      - prof: instructor id (kept for clustering)

    Rows with missing essential variables (eval, beauty, students, prof, age) are dropped.
    """
    df = df.copy()

    # Ensure numeric eval and beauty
    df['eval'] = pd.to_numeric(df['eval'], errors='coerce')
    df['beauty'] = pd.to_numeric(df['beauty'], errors='coerce')

    # Drop rows missing core variables
    df = df.dropna(subset=['eval', 'beauty', 'students', 'prof', 'age'])

    # Standardize beauty to create interpretable effect sizes
    beauty_mean = df['beauty'].mean()
    beauty_std = df['beauty'].std(ddof=0)
    # guard against zero std
    if beauty_std == 0 or np.isnan(beauty_std):
        df['beauty_z'] = df['beauty'] - beauty_mean
    else:
        df['beauty_z'] = (df['beauty'] - beauty_mean) / beauty_std

    # Map categorical controls to binary indicators (explicit, robust to capitalisation)
    df['gender_male'] = df['gender'].astype(str).str.strip().str.lower().map({'male': 1, 'female': 0})
    df['minority_yes'] = df['minority'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
    df['credits_single'] = df['credits'].astype(str).str.strip().str.lower().map({'single': 1, 'more': 0})
    df['division_lower'] = df['division'].astype(str).str.strip().str.lower().map({'lower': 1, 'upper': 0})
    df['native_yes'] = df['native'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
    df['tenure_yes'] = df['tenure'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})

    # If any of the mapped binary variables are NaN because of unexpected categories, fill with 0 and warn
    bin_cols = ['gender_male', 'minority_yes', 'credits_single', 'division_lower', 'native_yes', 'tenure_yes']
    for c in bin_cols:
        if df[c].isnull().any():
            # conservative choice: set missing binaries to 0 and preserve rows (alternatively could drop)
            df[c] = df[c].fillna(0)

    # Log-transform students to reduce skew
    df['students'] = pd.to_numeric(df['students'], errors='coerce')
    # Avoid taking log of zero or negative; those rows should already be dropped, but guard nonetheless
    df = df[df['students'] > 0]
    df['log_students'] = np.log(df['students'])

    # Keep only the columns required for modeling and drop any remaining rows with missing values
    out_cols = ['eval', 'beauty_z', 'age', 'gender_male', 'minority_yes', 'credits_single', 'division_lower', 'native_yes', 'tenure_yes', 'log_students', 'prof']
    df_out = df.loc[:, out_cols].dropna()

    # Ensure prof is integer (for clustering)
    try:
        df_out['prof'] = df_out['prof'].astype(int)
    except Exception:
        # if cannot cast, leave as-is (cluster code will still accept object groups)
        pass

    return df_out


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS model of student evaluations on instructor beauty and controls.

    Uses cluster-robust standard errors clustered by instructor (prof) to account for
    multiple course observations per instructor.

    Returns the fitted statsmodels results object (with clustered covariances).
    Also prints a short summary with coefficient, std err (clustered), t, and p-values.
    """
    import statsmodels.formula.api as smf

    # Formula: evaluation on standardized beauty and controls
    formula = (
        'eval ~ beauty_z + age + gender_male + minority_yes + credits_single + '
        'division_lower + native_yes + tenure_yes + log_students'
    )

    # Fit OLS
    mod = smf.ols(formula, data=df)
    # Use cluster-robust SEs by professor id
    try:
        res = mod.fit(cov_type='cluster', cov_kwds={'groups': df['prof']})
    except Exception:
        # fallback to heteroskedasticity-robust (HC3) if clustering fails
        res = mod.fit(cov_type='HC3')

    # Print concise summary
    print(res.summary())

    # Also return both the fitted result and a small table of the beauty coefficient and CI
    coef_table = res.get_robustcov_results().summary().tables[1] if hasattr(res, 'get_robustcov_results') else None

    return {
        'results': res,
        'formula': formula,
        'coef_beauty': res.params.get('beauty_z', None),
        'pvalue_beauty': res.pvalues.get('beauty_z', None),
        'conf_int_beauty': res.conf_int().loc['beauty_z'].tolist() if 'beauty_z' in res.params.index else None,
        'full_summary': res.summary()
    }


