from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/crofoot/shuffle_names_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw capuchin contest dataframe into the modeling dataframe.

    Produces the following new/guaranteed columns used in modeling:
      - dyad: binary outcome (1 focal won, 0 other won) [existing but enforced]
      - size_focal: total individuals in focal group (derived primarily from `f_other`, fallback to `focal`)
      - size_other: total individuals in other group (derived primarily from `f_focal`, fallback to `other`)
      - rel_size: size_focal - size_other
      - dist_from_home_focal: distance of focal group from center of its home range (from `win` column per schema)
      - dist_from_home_other: distance of other group from center of its home range (from `m_focal` column per schema)
      - rel_home_adv: dist_from_home_other - dist_from_home_focal (positive -> nearer focal home)
      - location_cat: categorical indicator ('FocalHome','OtherHome','Neutral') based on rel_home_adv
      - rel_size_z, rel_home_adv_z: z-scored versions for modeling

    The function also drops rows missing the core variables needed for analysis.
    """
    df = df.copy()

    # Ensure outcome is numeric and binary
    if 'dyad' not in df.columns:
        raise KeyError("Input dataframe must contain 'dyad' column (1 = focal won, 0 = other won).")
    df['dyad'] = pd.to_numeric(df['dyad'], errors='coerce')

    # Derive group size variables. According to the provided schema descriptions,
    # `f_other` is described as 'Number of individuals in focal group' and
    # `f_focal` as 'Number of individuals in other group'. These names are
    # somewhat confusing; use them as primary sources but fall back to `focal`
    # and `other` if needed.
    def safe_numeric(col):
        if col in df.columns:
            return pd.to_numeric(df[col], errors='coerce')
        else:
            return pd.Series(index=df.index, dtype='float64')

    f_other = safe_numeric('f_other')
    f_focal = safe_numeric('f_focal')
    focal_alt = safe_numeric('focal')
    other_alt = safe_numeric('other')

    # size_focal primarily from f_other, fallback to focal
    df['size_focal'] = f_other.fillna(focal_alt)
    # size_other primarily from f_focal, fallback to other
    df['size_other'] = f_focal.fillna(other_alt)

    # Distances from home-range centers. According to schema:
    #  - 'win' is distance of focal group from center of its home range
    #  - 'm_focal' is distance of other group from center of its home range
    df['dist_from_home_focal'] = safe_numeric('win')
    df['dist_from_home_other'] = safe_numeric('m_focal')

    # Number of males controls (keep as-is but ensure numeric)
    df['n_focal'] = safe_numeric('n_focal')
    df['n_other'] = safe_numeric('n_other')

    # Drop rows missing core required columns
    required = ['dyad', 'size_focal', 'size_other', 'dist_from_home_focal', 'dist_from_home_other']
    df = df.dropna(subset=required).reset_index(drop=True)

    # Compute relative size and relative home advantage
    df['rel_size'] = df['size_focal'] - df['size_other']
    df['rel_home_adv'] = df['dist_from_home_other'] - df['dist_from_home_focal']

    # Categorical location label for descriptive analyses
    df['location_cat'] = df['rel_home_adv'].apply(lambda x: 'FocalHome' if x > 0 else ('OtherHome' if x < 0 else 'Neutral'))

    # Standardize (z-score) the main continuous IVs for modeling stability and interpretability
    df['rel_size_z'] = (df['rel_size'] - df['rel_size'].mean()) / (df['rel_size'].std(ddof=0) if df['rel_size'].std(ddof=0) != 0 else 1.0)
    df['rel_home_adv_z'] = (df['rel_home_adv'] - df['rel_home_adv'].mean()) / (df['rel_home_adv'].std(ddof=0) if df['rel_home_adv'].std(ddof=0) != 0 else 1.0)

    # Final type enforcement
    df['dyad'] = df['dyad'].astype(int)
    df['n_focal'] = pd.to_numeric(df['n_focal'], errors='coerce')
    df['n_other'] = pd.to_numeric(df['n_other'], errors='coerce')

    # Return transformed dataframe that includes all columns needed for modeling
    return df

# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression (binomial GLM) predicting the probability the focal group wins (dyad == 1)
    from relative group size and contest location, including their interaction and controlling for male counts.

    Model formula (in matrix form):
      logit(P(dyad=1)) = b0 + b1*rel_size_z + b2*rel_home_adv_z + b3*(rel_size_z * rel_home_adv_z)
                          + b4*n_focal + b5*n_other

    Returns the fitted statsmodels results object.
    """
    df = df.copy()

    # Check required columns
    required = ['dyad', 'rel_size_z', 'rel_home_adv_z', 'n_focal', 'n_other']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for modeling: {missing}")

    # Design matrix
    df['interaction'] = df['rel_size_z'] * df['rel_home_adv_z']

    X = df[['rel_size_z', 'rel_home_adv_z', 'interaction', 'n_focal', 'n_other']].copy()
    X = sm.add_constant(X)
    y = df['dyad']

    # Fit a binomial GLM (logit link)
    model = sm.GLM(y, X, family=sm.families.Binomial())
    results = model.fit()

    # Return the fitted results object (user can call .summary() or access params)
    return results

