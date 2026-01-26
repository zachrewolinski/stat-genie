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
    Transform the raw Hamermesh classroom dataset into a modeling dataframe.
    Expected raw columns: feature1..feature13 as described in the schema.

    Produces columns used in modeling (see cvars):
      - Beauty, Eval, Age, Female, Minority, SingleCourse, UpperDivision,
        NativeEnglish, TenureTrack, NResponded, NEnrolled, ResponseRate,
        LogNEnrolled, InstructorID

    Behavior:
      - Rename features to semantic names
      - Convert categorical yes/no and factor values to binary indicators
      - Create response rate and log enrollment
      - Center Beauty (kept as 'Beauty' but also create Beauty_c if needed by user)
      - Drop rows with missing values in required columns
    """
    # Make a copy to avoid modifying in place
    df = df.copy()

    # Rename raw columns to semantic names
    rename_map = {
        'feature2': 'Minority_raw',
        'feature3': 'Age',
        'feature4': 'Gender_raw',
        'feature5': 'SingleCourse_raw',
        'feature6': 'Beauty',
        'feature7': 'Eval',
        'feature8': 'UpperDivision_raw',
        'feature9': 'NativeEnglish_raw',
        'feature10': 'TenureTrack_raw',
        'feature11': 'NResponded',
        'feature12': 'NEnrolled',
        'feature13': 'InstructorID'
    }
    df = df.rename(columns=rename_map)

    # Standardize string columns to lowercase where appropriate and strip whitespace
    for col in ['Minority_raw', 'Gender_raw', 'SingleCourse_raw', 'UpperDivision_raw', 'NativeEnglish_raw', 'TenureTrack_raw']:
        if col in df.columns:
            # convert non-null entries to str and lower
            df[col] = df[col].astype(str).str.strip().str.lower()

    # Map categorical raw values to binary indicators
    # Minority: 'yes' -> 1, else 0 (assumes 'no' or other values)
    df['Minority'] = df['Minority_raw'].map(lambda x: 1 if x == 'yes' else 0)

    # Gender: 'female' -> 1 else 0 (covers missing mapped to 0 after fill)
    df['Female'] = df['Gender_raw'].map(lambda x: 1 if x == 'female' else 0)

    # SingleCourse: 'single' -> 1 else 0
    df['SingleCourse'] = df['SingleCourse_raw'].map(lambda x: 1 if x == 'single' else 0)

    # UpperDivision: 'upper' -> 1 else 0
    df['UpperDivision'] = df['UpperDivision_raw'].map(lambda x: 1 if x == 'upper' else 0)

    # NativeEnglish: 'yes' -> 1 else 0
    df['NativeEnglish'] = df['NativeEnglish_raw'].map(lambda x: 1 if x == 'yes' else 0)

    # TenureTrack: 'yes' -> 1 else 0
    df['TenureTrack'] = df['TenureTrack_raw'].map(lambda x: 1 if x == 'yes' else 0)

    # Ensure numeric columns are numeric
    df['Beauty'] = pd.to_numeric(df['Beauty'], errors='coerce')
    df['Eval'] = pd.to_numeric(df['Eval'], errors='coerce')
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['NResponded'] = pd.to_numeric(df['NResponded'], errors='coerce')
    df['NEnrolled'] = pd.to_numeric(df['NEnrolled'], errors='coerce')
    df['InstructorID'] = pd.to_numeric(df['InstructorID'], errors='coerce')

    # Create response rate and log enrollment variables
    # Avoid division by zero
    df['ResponseRate'] = df.apply(lambda r: (r['NResponded'] / r['NEnrolled']) if (pd.notnull(r['NResponded']) and pd.notnull(r['NEnrolled']) and r['NEnrolled'] > 0) else np.nan, axis=1)
    df['LogNEnrolled'] = df['NEnrolled'].apply(lambda x: np.log(x) if pd.notnull(x) and x > 0 else np.nan)

    # Center the Beauty variable to improve interpretability of intercept and interactions
    if 'Beauty' in df.columns:
        beauty_mean = df['Beauty'].mean(skipna=True)
        df['Beauty'] = df['Beauty'] - beauty_mean

    # Select and keep only the final columns needed for modeling
    final_cols = [
        'Beauty', 'Eval', 'Age', 'Female', 'Minority', 'SingleCourse', 'UpperDivision',
        'NativeEnglish', 'TenureTrack', 'NResponded', 'NEnrolled', 'ResponseRate', 'LogNEnrolled', 'InstructorID'
    ]
    # If some expected columns are missing in the input, raise a clear error
    missing = [c for c in final_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing expected columns after transformation: {missing}")

    # Drop rows with missing values in the key modeling columns
    df = df.dropna(subset=['Beauty', 'Eval', 'NEnrolled', 'NResponded', 'InstructorID'])

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Run an OLS regression of Eval on Beauty and controls. Cluster robust SEs at the InstructorID level.

    Model specification:
      Eval ~ Beauty + Age + Female + Minority + SingleCourse + UpperDivision
             + NativeEnglish + TenureTrack + LogNEnrolled + ResponseRate
             + Beauty:Female

    Returns:
      fitted_results: statsmodels RegressionResults object (with cluster-robust cov)
    """
    import statsmodels.formula.api as smf

    # Ensure required columns are present
    required = ['Eval', 'Beauty', 'Age', 'Female', 'Minority', 'SingleCourse', 'UpperDivision',
                'NativeEnglish', 'TenureTrack', 'LogNEnrolled', 'ResponseRate', 'InstructorID']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for modeling: {missing}")

    # Build formula with interaction between Beauty and Female to test moderation by gender
    formula = (
        'Eval ~ Beauty + Age + Female + Minority + SingleCourse + UpperDivision '
        '+ NativeEnglish + TenureTrack + LogNEnrolled + ResponseRate + Beauty:Female'
    )

    # Fit OLS with cluster-robust standard errors by InstructorID
    model_fit = smf.ols(formula=formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['InstructorID']})

    # Return the fitted results object. Users can call .summary(), .params, .conf_int(), etc.
    return model_fit


