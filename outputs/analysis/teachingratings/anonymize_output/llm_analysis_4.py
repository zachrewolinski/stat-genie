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
    Transform the raw dataset (feature1...feature13) into a cleaned dataframe with
    columns used by the statistical model.

    Produced columns (exact names used by model):
      - RowID, Minority, Age, Gender, SingleCourse, Beauty, EvalScore,
        CourseLevel, NativeEnglish, TenureTrack, NumRespondents, Enrolled, InstructorID,
        ResponseRate, Beauty_z
    """
    df = df.copy()

    # Rename columns to meaningful names used downstream
    rename_map = {
        'feature1': 'RowID',
        'feature2': 'Minority',
        'feature3': 'Age',
        'feature4': 'Gender',
        'feature5': 'SingleCourse',
        'feature6': 'Beauty',
        'feature7': 'EvalScore',
        'feature8': 'CourseLevel',
        'feature9': 'NativeEnglish',
        'feature10': 'TenureTrack',
        'feature11': 'NumRespondents',
        'feature12': 'Enrolled',
        'feature13': 'InstructorID'
    }
    df = df.rename(columns=rename_map)

    # Ensure numeric columns are numeric and coerce bad values to NaN
    numeric_cols = ['Beauty', 'EvalScore', 'Age', 'NumRespondents', 'Enrolled', 'InstructorID']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows that cannot inform the main relationship (missing DV or main IV)
    df = df.dropna(subset=['EvalScore', 'Beauty']).reset_index(drop=True)

    # Standardize / normalize categorical text fields and coerce consistent labels
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].astype(str).str.strip().str.lower().replace({
            'male': 'Male', 'female': 'Female'
        })
    if 'Minority' in df.columns:
        df['Minority'] = df['Minority'].astype(str).str.strip().str.lower().replace({
            'yes': 'Yes', 'no': 'No'
        })
    if 'CourseLevel' in df.columns:
        df['CourseLevel'] = df['CourseLevel'].astype(str).str.strip().str.lower().replace({
            'lower': 'Lower', 'upper': 'Upper'
        })
    if 'NativeEnglish' in df.columns:
        df['NativeEnglish'] = df['NativeEnglish'].astype(str).str.strip().str.lower().replace({
            'yes': 'Yes', 'no': 'No'
        })
    if 'TenureTrack' in df.columns:
        df['TenureTrack'] = df['TenureTrack'].astype(str).str.strip().str.lower().replace({
            'yes': 'Yes', 'no': 'No'
        })
    # feature5 had values like 'single' or 'more' in the schema; normalize to Yes/No for single-credit
    if 'SingleCourse' in df.columns:
        df['SingleCourse'] = df['SingleCourse'].astype(str).str.strip().str.lower().replace({
            'single': 'Yes', 'yes': 'Yes', 'more': 'No', 'no': 'No'
        })

    # Construct response rate (safely) and clip to [0,1]
    df['ResponseRate'] = pd.to_numeric(df['NumRespondents'], errors='coerce') / pd.to_numeric(df['Enrolled'], errors='coerce')
    df['ResponseRate'] = df['ResponseRate'].fillna(0.0).clip(lower=0.0, upper=1.0)

    # Standardize the Beauty measure to a z-score (use population std to be explicit)
    beauty_mean = df['Beauty'].mean()
    beauty_std = df['Beauty'].std(ddof=0)
    # guard against zero std
    if beauty_std == 0 or np.isnan(beauty_std):
        df['Beauty_z'] = 0.0
    else:
        df['Beauty_z'] = (df['Beauty'] - beauty_mean) / beauty_std

    # Final housekeeping: ensure categorical dtypes where appropriate
    for cat in ['Gender', 'Minority', 'CourseLevel', 'NativeEnglish', 'TenureTrack', 'SingleCourse']:
        if cat in df.columns:
            df[cat] = df[cat].astype('category')

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fits an OLS regression of EvalScore on standardized Beauty and a set of controls.

    The model uses cluster-robust standard errors clustered by InstructorID to account for
    non-independence of observations from the same instructor.

    Formula used:
      EvalScore ~ Beauty_z + Age + C(Gender) + C(Minority) + C(CourseLevel) +
                  C(NativeEnglish) + C(TenureTrack) + C(SingleCourse) +
                  NumRespondents + Enrolled + ResponseRate

    Returns the fitted regression results object (statsmodels RegressionResultsWrapper).
    """
    # local import for formula API
    import statsmodels.formula.api as smf

    # Make a defensive copy
    df_model = df.copy()

    # Build formula. C(...) instructs statsmodels to treat those as categorical.
    formula = (
        'EvalScore ~ Beauty_z + Age + '
        'C(Gender) + C(Minority) + C(CourseLevel) + '
        'C(NativeEnglish) + C(TenureTrack) + C(SingleCourse) + '
        'NumRespondents + Enrolled + ResponseRate'
    )

    # Fit OLS with clustering of standard errors by InstructorID
    ols_mod = smf.ols(formula=formula, data=df_model)

    # If InstructorID contains missing or non-integer values, ensure groups are defined
    groups = df_model['InstructorID'] if 'InstructorID' in df_model.columns else None

    # Fit model and use cluster-robust SEs when InstructorID is available
    if groups is not None:
        results = ols_mod.fit(cov_type='cluster', cov_kwds={'groups': groups})
    else:
        results = ols_mod.fit()

    # Return the fitted results object (caller can inspect .summary(), .params, etc.)
    return results


