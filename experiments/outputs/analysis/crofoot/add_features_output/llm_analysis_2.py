from typing import Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/add_features_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset to create the variables used in the model.

    Adds:
    - size_diff: n_focal - n_other
    - size_ratio: n_focal / n_other (kept for diagnostics)
    - rel_dist: dist_other - dist_focal (positive => favorable for focal)
    - focal_home: binary indicator (1 if dist_focal < dist_other)
    - m_diff: m_focal - m_other
    - f_diff: f_focal - f_other
    - z standardized versions as helper columns: z_size_diff, z_rel_dist, z_m_diff, z_f_diff
    - ensures dyad is categorical

    Drops rows with missing values in required columns.
    """
    df = df.copy()

    # Required columns for the analysis
    required_cols = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other', 'm_focal', 'm_other', 'f_focal', 'f_other', 'dyad']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise KeyError(f"Missing required columns in input dataframe: {missing}")

    # Drop rows with NA in key columns
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    # Numeric transforms
    df['size_diff'] = df['n_focal'] - df['n_other']
    # Keep ratio for diagnostics if desired
    # Guard division by zero (shouldn't happen given schema, but be safe)
    df['size_ratio'] = df['n_focal'] / df['n_other'].replace(0, np.nan)

    # Location: positive means contest is deeper in focal's home-range (other is farther from its center)
    df['rel_dist'] = df['dist_other'] - df['dist_focal']
    # Binary home indicator (1 if closer to focal group's center)
    df['focal_home'] = (df['dist_focal'] < df['dist_other']).astype(int)

    # Sex composition differences
    df['m_diff'] = df['m_focal'] - df['m_other']
    df['f_diff'] = df['f_focal'] - df['f_other']

    # Standardize continuous predictors (z-scores) for diagnostics/model stability if desired
    def zscore(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        mu = s.mean()
        sd = s.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return s - mu
        return (s - mu) / sd

    # Helper standardized columns (allowed as internal helpers)
    df['z_size_diff'] = zscore(df['size_diff'])
    df['z_rel_dist'] = zscore(df['rel_dist'])
    df['z_m_diff'] = zscore(df['m_diff'])
    df['z_f_diff'] = zscore(df['f_diff'])

    # Make dyad a categorical column (keeps original values, but cast to category)
    df['dyad'] = df['dyad'].astype('category')

    # Ensure win is integer 0/1
    df['win'] = df['win'].astype(int)

    # Return transformed dataframe with all columns that may be used in modeling
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (binomial GLM) predicting probability that the focal group wins.

    Primary specification:
    - DV: win
    - IVs: size_diff (difference in group size), focal_home (binary moderator)
    - Interaction: size_diff * focal_home (tests whether home advantage modifies effect of size)
    - Additional covariates: rel_dist (continuous location measure), m_diff, f_diff
    - Controls: dyad fixed effects via C(dyad)

    Returns the fitted statsmodels result object (GLM).
    """
    # Check that required transformed columns exist
    required = ['win', 'size_diff', 'focal_home', 'rel_dist', 'm_diff', 'f_diff', 'dyad']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise KeyError(f"Missing required transformed columns for modeling: {missing}")

    # Formula: interaction between size_diff and focal_home (moderator),
    # plus continuous rel_dist and sex-composition controls and dyad fixed effects
    formula = 'win ~ size_diff * focal_home + rel_dist + m_diff + f_diff + C(dyad)'

    # Fit the logistic regression (binomial family) using statsmodels GLM.
    # Use families from statsmodels.api (imported as sm).
    result = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Return the fitted result object (has .summary(), .params, .predict, etc.)
    return result