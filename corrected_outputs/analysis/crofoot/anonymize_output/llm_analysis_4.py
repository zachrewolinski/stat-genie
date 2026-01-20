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
    Transform the raw dataset into a dataframe with named columns and derived predictors used in the model.

    Input columns in raw df are expected to include:
      feature1..feature12 as described in the schema.

    Output contains:
      - renamed columns for clarity
      - RelativeGroupSize, DistanceDifference, AtFocalHome
      - MaleDiff, FemaleDiff
      - z-scored versions of the main continuous predictors used in modeling
    """
    df = df.copy()

    # Rename raw columns to meaningful names
    rename_map = {
        'feature1': 'FocalGroupID',
        'feature2': 'OtherGroupID',
        'feature3': 'DyadID',
        'feature4': 'Win',
        'feature5': 'FocalDist',
        'feature6': 'OtherDist',
        'feature7': 'FocalSize',
        'feature8': 'OtherSize',
        'feature9': 'FocalMales',
        'feature10': 'OtherMales',
        'feature11': 'FocalFemales',
        'feature12': 'OtherFemales'
    }
    df = df.rename(columns=rename_map)

    # Convert key columns to numeric where appropriate
    numeric_cols = ['Win', 'FocalDist', 'OtherDist', 'FocalSize', 'OtherSize',
                    'FocalMales', 'OtherMales', 'FocalFemales', 'OtherFemales']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Drop rows with missing values in columns required for the analysis
    req_cols = ['Win', 'FocalDist', 'OtherDist', 'FocalSize', 'OtherSize',
                'FocalMales', 'OtherMales', 'FocalFemales', 'OtherFemales', 'DyadID']
    df = df.dropna(subset=req_cols).reset_index(drop=True)

    # Derived predictors
    # Relative group size (focal - other)
    df['RelativeGroupSize'] = df['FocalSize'] - df['OtherSize']
    # Distance difference: positive if contest is closer to focal group's home (OtherDist - FocalDist)
    df['DistanceDifference'] = df['OtherDist'] - df['FocalDist']
    # Binary indicator: contest closer to focal group's home-range center
    df['AtFocalHome'] = (df['FocalDist'] < df['OtherDist']).astype(int)

    # Sex-composition differences
    df['MaleDiff'] = df['FocalMales'] - df['OtherMales']
    df['FemaleDiff'] = df['FocalFemales'] - df['OtherFemales']

    # z-score continuous predictors for ease of interpretation and to stabilize estimation
    def zscore(series: pd.Series) -> pd.Series:
        if series.std(ddof=0) == 0 or np.isnan(series.std(ddof=0)):
            return (series - series.mean()) * 0.0
        return (series - series.mean()) / series.std(ddof=0)

    df['z_RelativeGroupSize'] = zscore(df['RelativeGroupSize'])
    df['z_DistanceDifference'] = zscore(df['DistanceDifference'])
    df['z_MaleDiff'] = zscore(df['MaleDiff'])
    df['z_FemaleDiff'] = zscore(df['FemaleDiff'])

    # Ensure DyadID and group IDs are treated as categorical strings for modeling
    df['DyadID'] = df['DyadID'].astype(str)
    df['FocalGroupID'] = df['FocalGroupID'].astype(str)
    df['OtherGroupID'] = df['OtherGroupID'].astype(str)

    # Keep only columns needed for modeling and interpretation (but leave originals for traceability)
    # Columns required by the model are: Win, z_RelativeGroupSize, z_DistanceDifference, z_MaleDiff, z_FemaleDiff, DyadID, and helper raw columns.
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression model predicting the probability that the focal group won (Win = 1).

    The primary model tests the main effects of relative group size and contest location and their interaction:
      Win ~ z_RelativeGroupSize * z_DistanceDifference

    Controls include sex-composition differences and dyad fixed effects (C(DyadID)).

    Returns the fitted GLM results object (binomial family / logit link).
    """
    import statsmodels.formula.api as smf

    df = df.copy()

    # Formula: include interaction between relative size and location; include sex-diff controls and dyad fixed effects
    formula = 'Win ~ z_RelativeGroupSize * z_DistanceDifference + z_MaleDiff + z_FemaleDiff + C(DyadID)'

    # Fit GLM with binomial family (logistic regression)
    results = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Return the fitted results object so the caller can inspect coefficients, summary, predictions, etc.
    return results


