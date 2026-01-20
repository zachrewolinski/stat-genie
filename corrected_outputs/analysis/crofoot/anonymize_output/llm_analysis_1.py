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
    Transform raw dataset columns into analysis-ready columns.

    Input expected columns (from schema):
      - feature1: focal group ID
      - feature2: other group ID
      - feature3: dyad ID
      - feature4: binary outcome (1 focal won, 0 other won)
      - feature5: focal distance from its home-range center (meters)
      - feature6: other distance from its home-range center (meters)
      - feature7: focal group size (count)
      - feature8: other group size (count)
      - feature9: focal number of males
      - feature10: other number of males
      - feature11: focal number of females
      - feature12: other number of females

    Returns dataframe containing the columns used in the model:
      - Win, LogSizeRatio, RelSize, AtFocalHome, DeltaMales, DeltaFemales, TotalSize, DyadID, FocalID, OtherID
    """
    df = df.copy()

    # 1) rename raw feature columns to meaningful analysis names
    mapping = {
        'feature1': 'FocalID',
        'feature2': 'OtherID',
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
    df = df.rename(columns=mapping)

    # 2) ensure numeric types where appropriate
    numeric_cols = ['Win', 'FocalDist', 'OtherDist', 'FocalSize', 'OtherSize', 'FocalMales', 'OtherMales', 'FocalFemales', 'OtherFemales']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3) drop rows missing any required values for the analysis
    required = ['Win', 'FocalDist', 'OtherDist', 'FocalSize', 'OtherSize', 'FocalMales', 'OtherMales', 'FocalFemales', 'OtherFemales', 'DyadID']
    df = df.dropna(subset=required)

    # 4) coerce binary outcome to int (0/1)
    df['Win'] = df['Win'].astype(int)

    # 5) derive relative-size measures
    # LogSizeRatio: log( (FocalSize + small) / (OtherSize + small) ) to avoid divide-by-zero
    small = 0.1
    df['LogSizeRatio'] = np.log((df['FocalSize'] + small) / (df['OtherSize'] + small))
    # RelSize: proportion of total that is focal (0-1)
    df['RelSize'] = df['FocalSize'] / (df['FocalSize'] + df['OtherSize'])

    # 6) derive composition differences and totals
    df['DeltaMales'] = df['FocalMales'] - df['OtherMales']
    df['DeltaFemales'] = df['FocalFemales'] - df['OtherFemales']
    df['TotalSize'] = df['FocalSize'] + df['OtherSize']

    # 7) derive contest location relative to home-range centers
    # If focal group's distance to its home center is less than or equal to the other group's distance to its own center,
    # consider contest to be at/closer to focal home range (AtFocalHome=1). Otherwise AtFocalHome=0.
    df['AtFocalHome'] = (df['FocalDist'] <= df['OtherDist']).astype(int)

    # 8) ensure ID columns are string for grouping/clustering
    df['DyadID'] = df['DyadID'].astype(str)
    df['FocalID'] = df['FocalID'].astype(str)
    df['OtherID'] = df['OtherID'].astype(str)

    # return only (or at least) the columns needed for modeling
    keep_cols = ['Win', 'LogSizeRatio', 'RelSize', 'AtFocalHome', 'DeltaMales', 'DeltaFemales', 'TotalSize', 'DyadID', 'FocalID', 'OtherID']
    for c in keep_cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression predicting the probability that the focal group wins.

    Model specification (primary):
      Win ~ LogSizeRatio * AtFocalHome + DeltaMales + DeltaFemales + TotalSize

    - LogSizeRatio: main independent variable (relative group size)
    - AtFocalHome: moderator (contest location: 1 = at focal home, 0 = at other/home-neutral)
    - Interaction tests whether the effect of relative size differs by location.

    Returns:
      A fitted statsmodels results object with cluster-robust SEs clustered on DyadID (if possible).
    """
    import statsmodels.formula.api as smf

    # Ensure required columns exist
    required = ['Win', 'LogSizeRatio', 'AtFocalHome', 'DeltaMales', 'DeltaFemales', 'TotalSize', 'DyadID']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # formula: include interaction between relative size and location
    formula = 'Win ~ LogSizeRatio * AtFocalHome + DeltaMales + DeltaFemales + TotalSize'

    # Fit logistic regression (binomial GLM) using statsmodels formula API
    # Use Logit for maximum-likelihood, then compute cluster-robust SEs by DyadID
    glm_model = smf.logit(formula=formula, data=df)
    fitted = glm_model.fit(disp=False)

    # Try to compute cluster-robust covariance by DyadID. If that fails, return the plain fit.
    try:
        clustered = fitted.get_robustcov_results(cov_type='cluster', groups=df['DyadID'])
        return clustered
    except Exception:
        # fallback: return the original fitted model
        return fitted


