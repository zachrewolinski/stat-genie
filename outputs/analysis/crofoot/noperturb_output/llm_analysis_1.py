from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/noperturb_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and derive variables needed for the analysis.

    Final dataframe columns used in the model:
      - win: binary outcome (0/1)
      - RelSize, RelSize_diff, RelSize_z: relative size measures and standardized ratio
      - Location: categorical location label ('FocalHome', 'OtherHome', 'Neutral')
      - Loc_FocalHome, Loc_OtherHome: dummy columns for Location (Neutral is reference)
      - RelMales, RelMales_z: male difference and standardized
      - RelDist, RelDist_z: continuous relative distance and standardized (dist_other - dist_focal)
    """
    df = df.copy()

    # Required columns
    required = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other', 'm_focal', 'm_other']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    # Drop rows with missing values in any required columns
    df = df.dropna(subset=required)

    # Ensure types
    df['win'] = df['win'].astype(int)
    numeric_cols = ['n_focal','n_other','dist_focal','dist_other','m_focal','m_other']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=numeric_cols + ['win'])

    # Relative size metrics
    # Ratio (preferred for multiplicative effects) and difference
    # Protect against division by zero, though n_other should be >=5 per schema
    df['RelSize'] = df['n_focal'] / df['n_other']
    df['RelSize_diff'] = df['n_focal'] - df['n_other']

    # Relative male counts
    df['RelMales'] = df['m_focal'] - df['m_other']

    # Relative distance: positive => focal is closer to contest location than other
    df['RelDist'] = df['dist_other'] - df['dist_focal']

    # Define categorical location. Use a small buffer (30 m) to define 'Neutral' band where contest is roughly equidistant.
    buffer = 30.0
    df['Location'] = 'Neutral'
    df.loc[df['dist_focal'] + buffer < df['dist_other'], 'Location'] = 'FocalHome'
    df.loc[df['dist_other'] + buffer < df['dist_focal'], 'Location'] = 'OtherHome'

    # Standardize continuous predictors (z-score). Use population std (ddof=0) to avoid small-sample ddof issues.
    for col in ['RelSize', 'RelSize_diff', 'RelMales', 'RelDist']:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            # If no variance, create a zero column
            df[col + '_z'] = 0.0
        else:
            df[col + '_z'] = (df[col] - mean) / std

    # Create dummy variables for Location. Keep dummies for FocalHome and OtherHome; Neutral will be the reference.
    loc_dummies = pd.get_dummies(df['Location'], prefix='Loc')
    # Ensure expected dummy column names exist even if zero in sample
    for cname in ['Loc_FocalHome', 'Loc_OtherHome', 'Loc_Neutral']:
        if cname not in loc_dummies.columns:
            loc_dummies[cname] = 0
    df = pd.concat([df, loc_dummies[['Loc_FocalHome', 'Loc_OtherHome', 'Loc_Neutral']]], axis=1)

    # For modeling, we'll use Loc_FocalHome and Loc_OtherHome and treat Loc_Neutral as reference.
    # Keep only the columns needed for the model (plus some extras for inspection).
    keep_cols = [
        'focal','other','dyad','win',
        'n_focal','n_other','RelSize','RelSize_diff','RelSize_z',
        'm_focal','m_other','RelMales','RelMales_z',
        'dist_focal','dist_other','RelDist','RelDist_z',
        'Location','Loc_FocalHome','Loc_OtherHome','Loc_Neutral'
    ]
    # Some of these columns exist in the original df; filter to those present
    keep_cols = [c for c in keep_cols if c in df.columns]

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression predicting probability that focal group wins (win==1).

    Model specification (main analysis):
      logit( P(win=1) ) = b0 + b1*RelSize_z + b2*Loc_FocalHome + b3*Loc_OtherHome
                         + b4*(RelSize_z * Loc_FocalHome) + b5*(RelSize_z * Loc_OtherHome)
                         + b6*RelMales_z + b7*RelDist_z

    Location 'Neutral' is the omitted reference category. Interactions test whether effect of relative size differs across locations.

    Returns the fitted statsmodels results object for the Logit model.
    """
    df = df.copy()

    # Check required columns
    required = ['win','RelSize_z','RelMales_z','RelDist_z','Loc_FocalHome','Loc_OtherHome']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Transformed dataframe is missing columns required for modeling: {missing}")

    # Ensure dummies are numeric
    df['Loc_FocalHome'] = pd.to_numeric(df['Loc_FocalHome'], errors='coerce').fillna(0).astype(float)
    df['Loc_OtherHome'] = pd.to_numeric(df['Loc_OtherHome'], errors='coerce').fillna(0).astype(float)

    # Interaction terms
    df['RelSize_z_x_Loc_FocalHome'] = df['RelSize_z'] * df['Loc_FocalHome']
    df['RelSize_z_x_Loc_OtherHome'] = df['RelSize_z'] * df['Loc_OtherHome']

    # Design matrix
    X_cols = [
        'RelSize_z',
        'Loc_FocalHome',
        'Loc_OtherHome',
        'RelSize_z_x_Loc_FocalHome',
        'RelSize_z_x_Loc_OtherHome',
        'RelMales_z',
        'RelDist_z'
    ]
    X = df[X_cols].astype(float)
    X = sm.add_constant(X, has_constant='add')

    y = df['win'].astype(float)

    # Fit logistic regression (maximum likelihood)
    logit_model = sm.Logit(y, X)
    try:
        results = logit_model.fit(disp=False)
    except Exception as e:
        # If convergence issues occur, try with method='newton' and more iterations
        results = logit_model.fit(disp=False, method='newton', maxiter=200)

    return results


