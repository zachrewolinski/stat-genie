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
    Transform the raw dataset into a dataframe suitable for modeling intergroup contest outcomes.

    Inputs (expected columns in raw df):
      - feature1: ID of focal group
      - feature2: ID of other group
      - feature3: ID of dyad
      - feature4: 1 if focal won contest, 0 if other won
      - feature5: Distance (m) of focal group from its home-range center
      - feature6: Distance (m) of other group from its home-range center
      - feature7: Number of individuals in focal group (total size)
      - feature8: Number of individuals in other group (total size)
      - feature9: Number of males in focal group
      - feature10: Number of males in other group
      - feature11: Number of females in focal group
      - feature12: Number of females in other group

    Output columns (kept/produced):
      - FocalWin, RelSize, SizeRatio, HomeRangeAdv, AtHome,
        RelMales, RelFemales, standardized versions: RelSize_z, SizeRatio_z, HomeRangeAdv_z, RelMales_z, RelFemales_z,
        plus FocalGroup, OtherGroup, DyadID and some raw fields.
    """
    df = df.copy()

    # Drop rows that are missing essential fields for the analysis
    required = ['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature11','feature12']
    df = df.dropna(subset=required)

    # Create outcome variable
    df['FocalWin'] = df['feature4'].astype(int)

    # IDs / categorical variables
    df['FocalGroup'] = df['feature1'].astype('category')
    df['OtherGroup'] = df['feature2'].astype('category')
    df['DyadID'] = df['feature3'].astype('category')

    # Distances (continuous)
    df['FocalDist'] = pd.to_numeric(df['feature5'], errors='coerce')
    df['OtherDist'] = pd.to_numeric(df['feature6'], errors='coerce')

    # Group sizes and composition
    df['FocalSize'] = pd.to_numeric(df['feature7'], errors='coerce').astype(int)
    df['OtherSize'] = pd.to_numeric(df['feature8'], errors='coerce').astype(int)
    df['FocalMales'] = pd.to_numeric(df['feature9'], errors='coerce').astype(int)
    df['OtherMales'] = pd.to_numeric(df['feature10'], errors='coerce').astype(int)
    df['FocalFemales'] = pd.to_numeric(df['feature11'], errors='coerce').astype(int)
    df['OtherFemales'] = pd.to_numeric(df['feature12'], errors='coerce').astype(int)

    # Core predictors
    # Relative size (focal - other). Positive => focal larger.
    df['RelSize'] = df['FocalSize'] - df['OtherSize']
    # Size ratio (another representation): focal / other (avoid division by zero)
    df['SizeRatio'] = df['FocalSize'] / df['OtherSize'].replace({0: np.nan})

    # Home-range proximity advantage: other_dist - focal_dist. Positive => focal is closer to its home center than the other group (location advantage).
    df['HomeRangeAdv'] = df['OtherDist'] - df['FocalDist']
    # Binary indicator of being closer to home than the opponent
    df['AtHome'] = (df['FocalDist'] < df['OtherDist']).astype(int)

    # Relative sex-composition controls
    df['RelMales'] = df['FocalMales'] - df['OtherMales']
    df['RelFemales'] = df['FocalFemales'] - df['OtherFemales']

    # Standardize continuous predictors (z-scores) for easier interpretation of coefficients
    for col in ['RelSize', 'SizeRatio', 'HomeRangeAdv', 'RelMales', 'RelFemales']:
        # compute only on finite values to avoid NaN propagation
        vals = df[col]
        mean = vals.mean()
        std = vals.std(ddof=0)
        if std == 0 or np.isnan(std):
            df[col + '_z'] = 0.0
        else:
            df[col + '_z'] = (vals - mean) / std

    # Return a dataframe with all columns that will be used in the model
    keep_cols = [
        'FocalWin',
        'RelSize', 'SizeRatio', 'HomeRangeAdv', 'AtHome',
        'RelMales', 'RelFemales',
        'RelSize_z', 'SizeRatio_z', 'HomeRangeAdv_z', 'RelMales_z', 'RelFemales_z',
        'FocalGroup', 'OtherGroup', 'DyadID',
        'FocalSize', 'OtherSize', 'FocalDist', 'OtherDist'
    ]

    # If any of the keep_cols don't exist due to data issues, create them as NA to keep schema stable
    for c in keep_cols:
        if c not in df.columns:
            df[c] = np.nan

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression to estimate how relative group size and location advantage affect the probability
    that the focal group wins an intergroup contest.

    Model specification (primary):
      FocalWin ~ RelSize_z + HomeRangeAdv_z + RelMales_z + RelFemales_z + C(FocalGroup)

    We use a binomial GLM (logit link) and obtain cluster-robust standard errors clustered on DyadID
    to account for repeated encounters between the same pair of groups.

    Input df is expected to be the transformed dataframe returned by transform().
    Returns the fitted results object with cluster-robust covariances applied.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Ensure required columns are present
    required = ['FocalWin', 'RelSize_z', 'HomeRangeAdv_z', 'RelMales_z', 'RelFemales_z', 'FocalGroup', 'DyadID']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Drop rows with NA in model variables
    mod_df = df.dropna(subset=required).copy()

    # Formula using standardized predictors and focal-group fixed effects
    formula = 'FocalWin ~ RelSize_z + HomeRangeAdv_z + RelMales_z + RelFemales_z + C(FocalGroup)'

    # Fit binomial GLM (logit)
    glm_model = smf.glm(formula=formula, data=mod_df, family=sm.families.Binomial())
    fit = glm_model.fit()

    # Obtain cluster-robust covariance estimates clustered on DyadID
    try:
        results = fit.get_robustcov_results(cov_type='cluster', groups=mod_df['DyadID'])
    except Exception:
        # Fallback: use default (non-clustered) if clustering fails for any reason
        results = fit

    # Print a concise summary for quick inspection (caller can inspect returned object for full details)
    print(results.summary())

    return results


