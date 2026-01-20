from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/crofoot/anonymize_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset (with columns feature1..feature12) into a dataframe ready for modeling.

    Created columns (used in modeling):
      - FocalWin: binary outcome (from feature4)
      - SizeDiff: focal size - other size (from feature7, feature8)
      - SizeDiff_z: standardized SizeDiff (z-score)
      - DistDiff: focal distance - other distance (feature5 - feature6)
      - DistDiff_z: standardized DistDiff (z-score)
      - ContestLocation: categorical location label ('FocalHome','OtherHome','Neutral')
      - ContestLoc_OtherHome, ContestLoc_Neutral: dummy columns for ContestLocation (FocalHome is baseline)
      - FocalMales, OtherMales, FocalFemales, OtherFemales: counts from features 9-12

    Note: uses a threshold of 50 meters to classify 'home' vs 'neutral' contests. This threshold is adjustable.
    """
    df = df.copy()

    # Keep only rows with required fields (outcome, sizes, distances, composition)
    required = ['feature4', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9', 'feature10', 'feature11', 'feature12']
    df = df.dropna(subset=required)

    # Outcome
    df['FocalWin'] = df['feature4'].astype(int)

    # Distances: focal and other
    df['FocalDist'] = df['feature5'].astype(float)
    df['OtherDist'] = df['feature6'].astype(float)
    df['DistDiff'] = df['FocalDist'] - df['OtherDist']

    # Group sizes
    df['FocalSize'] = df['feature7'].astype(float)
    df['OtherSize'] = df['feature8'].astype(float)
    df['SizeDiff'] = df['FocalSize'] - df['OtherSize']

    # Standardize continuous predictors (z-scores)
    df['SizeDiff_z'] = (df['SizeDiff'] - df['SizeDiff'].mean()) / (df['SizeDiff'].std(ddof=0) if df['SizeDiff'].std(ddof=0) != 0 else 1.0)
    df['DistDiff_z'] = (df['DistDiff'] - df['DistDiff'].mean()) / (df['DistDiff'].std(ddof=0) if df['DistDiff'].std(ddof=0) != 0 else 1.0)

    # Contest location: classify relative to home-range centers
    # If focal much closer (DistDiff <= -50) -> FocalHome
    # If other much closer (DistDiff >= 50) -> OtherHome
    # Else -> Neutral
    threshold = 50.0
    def classify_location(d):
        if d <= -threshold:
            return 'FocalHome'
        elif d >= threshold:
            return 'OtherHome'
        else:
            return 'Neutral'
    df['ContestLocation'] = df['DistDiff'].apply(classify_location)

    # Convert to categorical with explicit ordering so get_dummies has consistent column names
    df['ContestLocation'] = pd.Categorical(df['ContestLocation'], categories=['FocalHome', 'OtherHome', 'Neutral'], ordered=False)
    contest_dummies = pd.get_dummies(df['ContestLocation'], prefix='ContestLoc', drop_first=True)
    # This produces columns ContestLoc_OtherHome and ContestLoc_Neutral (FocalHome is baseline)
    for c in contest_dummies.columns:
        df[c] = contest_dummies[c]

    # Composition controls
    df['FocalMales'] = df['feature9'].astype(float)
    df['OtherMales'] = df['feature10'].astype(float)
    df['FocalFemales'] = df['feature11'].astype(float)
    df['OtherFemales'] = df['feature12'].astype(float)

    # Keep only columns needed for modeling (plus some useful originals)
    keep_cols = [
        'FocalWin',
        'SizeDiff', 'SizeDiff_z',
        'DistDiff', 'DistDiff_z',
        'ContestLocation', 'ContestLoc_OtherHome', 'ContestLoc_Neutral',
        'FocalMales', 'OtherMales', 'FocalFemales', 'OtherFemales'
    ]
    # If any of the dummy columns don't exist (e.g., no observations in that category), create them with zeros
    for col in ['ContestLoc_OtherHome', 'ContestLoc_Neutral']:
        if col not in df.columns:
            df[col] = 0
            keep_cols.append(col)

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression predicting the probability that the focal group wins.

    Model specification (primary):
      FocalWin ~ SizeDiff_z + ContestLocation (dummies) + SizeDiff_z x ContestLocation + DistDiff_z + composition controls

    Interaction tests whether the effect of relative group size differs by contest location.

    Returns:
      - fitted GLM results object (Binomial family)
    """
    df = df.copy()

    # Build interaction terms between SizeDiff_z and contest-location dummies
    df['Size_x_OtherHome'] = df['SizeDiff_z'] * df['ContestLoc_OtherHome']
    df['Size_x_Neutral'] = df['SizeDiff_z'] * df['ContestLoc_Neutral']

    # Design matrix
    X_cols = [
        'SizeDiff_z',
        'ContestLoc_OtherHome',
        'ContestLoc_Neutral',
        'Size_x_OtherHome',
        'Size_x_Neutral',
        'DistDiff_z',
        'FocalMales',
        'OtherMales',
        'FocalFemales',
        'OtherFemales'
    ]

    # Ensure columns exist (in case some categories were absent and dummies are zeros)
    for col in X_cols:
        if col not in df.columns:
            df[col] = 0.0

    X = df[X_cols]
    X = sm.add_constant(X, has_constant='add')
    y = df['FocalWin']

    # Fit binomial GLM (logistic regression)
    model_fit = sm.GLM(y, X, family=sm.families.Binomial()).fit()

    # Return the fitted model object; users can inspect model_fit.summary()
    return model_fit


