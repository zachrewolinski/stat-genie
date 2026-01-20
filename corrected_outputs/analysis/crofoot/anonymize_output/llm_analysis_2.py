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
    - Rename raw feature columns to meaningful names.
    - Drop rows with missing values in critical fields.
    - Create derived variables used in modeling:
        * SizeRatio = SizeFocal / SizeOther
        * SizeDiff = SizeFocal - SizeOther
        * DistanceDiff = DistOtherHome - DistFocalHome (positive => contest closer to focal)
        * InFocalHome = 1 if DistFocalHome < DistOtherHome else 0
        * MalesDiff = MalesFocal - MalesOther
        * FemalesDiff = FemalesFocal - FemalesOther
    - Return only the columns needed for the model.
    """
    df = df.copy()

    # Rename columns to meaningful names
    df = df.rename(columns={
        'feature1': 'FocalGroup',
        'feature2': 'OtherGroup',
        'feature3': 'DyadID',
        'feature4': 'FocalWon',
        'feature5': 'DistFocalHome',
        'feature6': 'DistOtherHome',
        'feature7': 'SizeFocal',
        'feature8': 'SizeOther',
        'feature9': 'MalesFocal',
        'feature10': 'MalesOther',
        'feature11': 'FemalesFocal',
        'feature12': 'FemalesOther'
    })

    # drop rows missing critical fields
    required = ['FocalWon', 'DistFocalHome', 'DistOtherHome', 'SizeFocal', 'SizeOther', 'MalesFocal', 'MalesOther', 'FemalesFocal', 'FemalesOther']
    df = df.dropna(subset=required)

    # Ensure binary outcome is integer 0/1
    df['FocalWon'] = df['FocalWon'].astype(int)

    # Derived variables
    # Size ratio (handle division by zero)
    df['SizeRatio'] = df['SizeFocal'] / df['SizeOther']
    df.loc[~np.isfinite(df['SizeRatio']), 'SizeRatio'] = np.nan

    df['SizeDiff'] = df['SizeFocal'] - df['SizeOther']

    # Distance difference: positive when contest is relatively closer to focal group's home
    df['DistanceDiff'] = df['DistOtherHome'] - df['DistFocalHome']

    # Binary indicator for contest being in/closer to focal's home
    df['InFocalHome'] = (df['DistFocalHome'] < df['DistOtherHome']).astype(int)

    # Sex composition differences
    df['MalesDiff'] = df['MalesFocal'] - df['MalesOther']
    df['FemalesDiff'] = df['FemalesFocal'] - df['FemalesOther']

    # Drop rows with any newly introduced NA values (e.g., division by zero)
    df = df.dropna(subset=['SizeRatio', 'DistanceDiff', 'MalesDiff', 'FemalesDiff'])

    # Return only columns required for modeling (and identifiers for possible further use)
    out_cols = ['FocalWon', 'SizeRatio', 'SizeDiff', 'DistanceDiff', 'InFocalHome', 'MalesDiff', 'FemalesDiff', 'FocalGroup', 'OtherGroup', 'DyadID']
    return df[out_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    - Fit a logistic regression (binomial GLM with logit link) predicting the probability that the focal group won.
    - Predictors: SizeRatio (main effect), InFocalHome (main effect), their interaction, DistanceDiff (continuous location measure), MalesDiff and FemalesDiff (controls), and fixed effects for focal and other group IDs.
    - Returns the fitted statsmodels results object.

    Notes:
    - With only 58 observations and small number of groups, interpret group fixed effects cautiously.
    - Optionally, the user can compute cluster-robust standard errors by dyad (not implemented here but DyadID is available in the transformed df).
    """
    df = df.copy()

    # Interaction term: does location moderate the effect of size advantage?
    df['SizeRatio_InFocalHome'] = df['SizeRatio'] * df['InFocalHome']

    # Create fixed-effect dummies for FocalGroup and OtherGroup (drop first to avoid perfect multicollinearity)
    fg_dummies = pd.get_dummies(df['FocalGroup'].astype(str), prefix='FocalGroup', drop_first=True)
    og_dummies = pd.get_dummies(df['OtherGroup'].astype(str), prefix='OtherGroup', drop_first=True)

    # Design matrix
    X = pd.concat([
        df[['SizeRatio', 'InFocalHome', 'SizeRatio_InFocalHome', 'MalesDiff', 'FemalesDiff', 'DistanceDiff']],
        fg_dummies,
        og_dummies
    ], axis=1)

    # Add intercept
    X = sm.add_constant(X, has_constant='add')

    y = df['FocalWon']

    # Fit binomial GLM (logistic regression)
    model = sm.GLM(y, X, family=sm.families.Binomial()).fit()

    # Return the fitted model object (results)
    return model


