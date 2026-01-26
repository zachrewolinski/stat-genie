from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/teachingratings/anonymize_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset (feature1..feature13) into analysis-ready dataframe with named columns used in the model.
    Output columns (kept/created):
      - beauty_z: standardized beauty rating (z-score)
      - teaching_eval: course overall teaching evaluation (DV)
      - age_c, age_sq: centered age and squared term
      - female, minority, upper_division, single_credit, native_english, tenure_track: binary controls
      - participants, enrolled, response_rate, log_enrolled
      - instructor_id
    """
    df = df.copy()

    # Rename known columns to meaningful names
    rename_map = {
        'feature6': 'beauty',           # beauty rating (mean-zero originally)
        'feature7': 'teaching_eval',    # outcome
        'feature3': 'age',
        'feature4': 'gender',
        'feature2': 'minority_flag',    # 'yes'/'no'
        'feature5': 'single_credit_flag',
        'feature8': 'division',         # 'upper'/'lower'
        'feature9': 'native_english_flag',
        'feature10': 'tenure_flag',
        'feature11': 'participants',    # number participated in evaluation
        'feature12': 'enrolled',        # number enrolled
        'feature13': 'instructor_id'
    }
    df = df.rename(columns=rename_map)

    # Keep only rows with non-missing DV and IV
    df = df.dropna(subset=['beauty', 'teaching_eval'])

    # Convert numeric columns to proper dtypes
    for col in ['beauty', 'teaching_eval', 'age', 'participants', 'enrolled']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Binary / categorical processing (robust to capitalization)
    def safe_eq(series, val):
        return series.astype(str).str.lower().eq(str(val).lower())

    # Gender: feature4 values are 'male'/'female'
    df['female'] = (safe_eq(df.get('gender', pd.Series(['']*len(df))), 'female')).astype(int)

    # Minority: feature2 'yes' -> 1
    df['minority'] = (safe_eq(df.get('minority_flag', pd.Series(['']*len(df))), 'yes')).astype(int)

    # Single-credit elective: feature5 values 'single' indicates single-credit elective
    df['single_credit'] = (safe_eq(df.get('single_credit_flag', pd.Series(['']*len(df))), 'single')).astype(int)

    # Division: 'upper' -> 1 else 0
    df['upper_division'] = (safe_eq(df.get('division', pd.Series(['']*len(df))), 'upper')).astype(int)

    # Native English: feature9 'yes'
    df['native_english'] = (safe_eq(df.get('native_english_flag', pd.Series(['']*len(df))), 'yes')).astype(int)

    # Tenure track: feature10 'yes'
    df['tenure_track'] = (safe_eq(df.get('tenure_flag', pd.Series(['']*len(df))), 'yes')).astype(int)

    # Participants / enrolled: ensure numeric and reasonable
    df['participants'] = pd.to_numeric(df['participants'], errors='coerce')
    df['enrolled'] = pd.to_numeric(df['enrolled'], errors='coerce')

    # Remove rows with nonpositive enrollment (can't compute response rate)
    df = df[df['enrolled'].notna() & (df['enrolled'] > 0)]

    # Response rate and log enrollment
    df['response_rate'] = (df['participants'] / df['enrolled']).clip(lower=0, upper=1)
    df['log_enrolled'] = np.log(df['enrolled'] + 1)

    # Age: center and squared
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        age_mean = df['age'].mean()
        df['age_c'] = df['age'] - age_mean
        df['age_sq'] = df['age_c'] ** 2
    else:
        df['age_c'] = np.nan
        df['age_sq'] = np.nan

    # Standardize beauty (z-score) for interpretability. Use sample std (ddof=0) for population-like scaling.
    beauty_mean = df['beauty'].mean()
    beauty_std = df['beauty'].std(ddof=0)
    if beauty_std == 0 or np.isnan(beauty_std):
        # fallback: keep raw beauty if no variation
        df['beauty_z'] = df['beauty'] - beauty_mean
    else:
        df['beauty_z'] = (df['beauty'] - beauty_mean) / beauty_std

    # Ensure instructor_id is present and numeric or string
    if 'instructor_id' in df.columns:
        df['instructor_id'] = df['instructor_id'].astype(object)
    else:
        df['instructor_id'] = np.nan

    # Final set of columns to return (these are used in the modeling stage)
    keep_cols = [
        'beauty_z', 'teaching_eval', 'age_c', 'age_sq', 'female', 'minority',
        'upper_division', 'single_credit', 'native_english', 'tenure_track',
        'log_enrolled', 'response_rate', 'participants', 'enrolled', 'instructor_id'
    ]

    # If any of the keep_cols are missing (rare), create them with NA to keep consistent schema
    for c in keep_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Return only the columns needed for modeling
    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """ 
    Fit OLS regression of teaching evaluation on standardized beauty with controls.
    Uses cluster-robust standard errors clustered on instructor_id.

    Model specification:
      teaching_eval ~ beauty_z + age_c + age_sq + female + minority
                      + upper_division + single_credit + native_english + tenure_track
                      + log_enrolled + response_rate

    Returns the fitted statsmodels regression result object.
    """
    # Drop rows with missing values in key model columns
    model_cols = [
        'teaching_eval', 'beauty_z', 'age_c', 'age_sq', 'female', 'minority',
        'upper_division', 'single_credit', 'native_english', 'tenure_track',
        'log_enrolled', 'response_rate', 'instructor_id'
    ]
    df_model = df.dropna(subset=['teaching_eval', 'beauty_z'])  # always require DV and IV

    # We allow some missing controls; drop rows missing any control used in formula
    df_model = df_model.dropna(subset=[c for c in model_cols if c in df_model.columns])

    # Define formula
    formula = (
        'teaching_eval ~ beauty_z + age_c + age_sq + female + minority '
        '+ upper_division + single_credit + native_english + tenure_track '
        '+ log_enrolled + response_rate'
    )

    # Fit OLS and use clustering by instructor_id for robust SEs if instructor_id is available
    try:
        res = smf.ols(formula=formula, data=df_model).fit(
            cov_type='cluster', cov_kwds={'groups': df_model['instructor_id']}
        )
    except Exception:
        # If clustering fails (e.g., instructor_id constant), fall back to robust (HC3) SEs
        res = smf.ols(formula=formula, data=df_model).fit(cov_type='HC3')

    # Return the fitted result object (has .summary(), .params, .bse, etc.)
    return res