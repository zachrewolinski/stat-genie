from typing import Any, Dict, FrozenSet, List, Literal, Optional, Set, Tuple
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
    Prepare the dataset for modeling. The returned dataframe contains the following columns used in the model:
      - eval: dependent variable (course evaluation score)
      - beauty_z: standardized beauty rating
      - age: instructor age
      - gender_female, minority_yes, credits_more, division_upper, native_yes, tenure_yes: binary dummies
      - log_students: log(number of students who participated)
      - prof: instructor id (kept as-is for clustering / fixed effects)

    This function drops rows with missing values in variables required for the analysis.
    """

    # Ensure a copy
    df = df.copy()

    # Required raw columns
    required_cols = ['beauty', 'eval', 'age', 'gender', 'minority', 'credits', 'division', 'native', 'tenure', 'students', 'prof']

    # Drop rows missing any of the required columns
    df = df.dropna(subset=required_cols)

    # Ensure numeric types where expected
    df['eval'] = pd.to_numeric(df['eval'], errors='coerce')
    df['beauty'] = pd.to_numeric(df['beauty'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['students'] = pd.to_numeric(df['students'], errors='coerce')
    # Keep prof numeric as required; if non-numeric, coerce to numeric (may produce NaN which will be dropped)
    df['prof'] = pd.to_numeric(df['prof'], errors='coerce')

    # Drop rows that became NA after coercion
    df = df.dropna(subset=['eval', 'beauty', 'age', 'students', 'prof'])

    # Standardize beauty (z-score)
    beauty_mean = df['beauty'].mean()
    beauty_std = df['beauty'].std(ddof=0) if df['beauty'].std(ddof=0) != 0 else 1.0
    df['beauty_z'] = (df['beauty'] - beauty_mean) / beauty_std

    # Binary dummies for categorical controls (explicit columns used in modeling)
    # gender: 'male' / 'female'
    df['gender_female'] = (df['gender'].astype(str).str.lower() == 'female').astype(int)

    # minority: 'yes' / 'no'
    df['minority_yes'] = (df['minority'].astype(str).str.lower() == 'yes').astype(int)

    # credits: 'single' / 'more'
    df['credits_more'] = (df['credits'].astype(str).str.lower() == 'more').astype(int)

    # division: 'lower' / 'upper'
    df['division_upper'] = (df['division'].astype(str).str.lower() == 'upper').astype(int)

    # native: 'yes' / 'no'
    df['native_yes'] = (df['native'].astype(str).str.lower() == 'yes').astype(int)

    # tenure: 'yes' / 'no'
    df['tenure_yes'] = (df['tenure'].astype(str).str.lower() == 'yes').astype(int)

    # Transform students -> log_students to reduce skew
    # Add small constant in case of zeros (dataset min students >= 5 so safe, but keep as precaution)
    df['log_students'] = np.log(df['students'].clip(lower=1))

    # Final drop: ensure no missing in derived columns
    model_cols = ['eval', 'beauty_z', 'age', 'gender_female', 'minority_yes', 'credits_more', 'division_upper', 'native_yes', 'tenure_yes', 'log_students', 'prof']
    df = df.dropna(subset=model_cols)

    # Reset index for a clean returned dataframe
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Runs two specifications to estimate the effect of instructor beauty on student evaluations:
      1) Baseline OLS with clustered (by prof) robust standard errors.
      2) OLS with professor fixed effects (C(prof)) and clustered SEs by prof.

    Returns a dictionary with fitted results objects and summary text for convenience.
    """

    results: Dict[str, Any] = {}

    # Formula for baseline controls
    formula_base = ('eval ~ beauty_z + age + gender_female + minority_yes + credits_more '
                    '+ division_upper + native_yes + tenure_yes + log_students')

    # 1) Baseline OLS
    ols_base = smf.ols(formula_base, data=df).fit()
    # Clustered SEs by professor id
    try:
        ols_base_clust = ols_base.get_robustcov_results(cov_type='cluster', groups=df['prof'])
    except Exception:
        # fallback to heteroskedasticity-robust (HC1) if clustering fails
        ols_base_clust = ols_base.get_robustcov_results(cov_type='HC1')

    results['baseline_model'] = ols_base_clust
    results['baseline_summary'] = ols_base_clust.summary().as_text()

    # 2) Professor fixed effects (adds C(prof)). Note: this will soak up between-professor variation.
    formula_fe = formula_base + ' + C(prof)'
    ols_fe = smf.ols(formula_fe, data=df).fit()
    try:
        ols_fe_clust = ols_fe.get_robustcov_results(cov_type='cluster', groups=df['prof'])
    except Exception:
        ols_fe_clust = ols_fe.get_robustcov_results(cov_type='HC1')

    results['prof_fe_model'] = ols_fe_clust
    results['prof_fe_summary'] = ols_fe_clust.summary().as_text()

    # Helper to robustly extract parameter and standard error by name,
    # handling both pandas Series and numpy arrays for params/bse.
    def _get_coef_and_se(res, param_name: str):
        coef = np.nan
        se = np.nan
        params = getattr(res, 'params', None)
        bse = getattr(res, 'bse', None)

        if params is None or bse is None:
            return coef, se

        # Try dict-like access first (pandas Series)
        try:
            coef = float(params.get(param_name, np.nan))
        except Exception:
            # params may be ndarray; try to locate index via model exog names
            try:
                exog_names = list(getattr(res.model, 'exog_names', []))
                if param_name in exog_names:
                    idx = exog_names.index(param_name)
                    coef = float(params[idx])
                else:
                    coef = np.nan
            except Exception:
                coef = np.nan

        try:
            se = float(bse.get(param_name, np.nan))
        except Exception:
            try:
                exog_names = list(getattr(res.model, 'exog_names', []))
                if param_name in exog_names:
                    idx = exog_names.index(param_name)
                    se = float(bse[idx])
                else:
                    se = np.nan
            except Exception:
                se = np.nan

        return coef, se

    # Complementary check: magnitude and standardized effect
    # Compute coefficient and 95% CI for beauty_z from baseline
    coef, se = _get_coef_and_se(ols_base_clust, 'beauty_z')
    ci_lower = coef - 1.96 * se if not np.isnan(coef) and not np.isnan(se) else (np.nan)
    ci_upper = coef + 1.96 * se if not np.isnan(coef) and not np.isnan(se) else (np.nan)
    results['beauty_effect_baseline'] = {
        'coef': coef,
        'se': se,
        '95ci': (ci_lower, ci_upper)
    }

    # Return results dictionary (contains model result objects and human-readable summaries)
    return results