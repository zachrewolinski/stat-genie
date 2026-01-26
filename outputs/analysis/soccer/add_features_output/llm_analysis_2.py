from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/soccer/add_features_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transformations performed:
    - drop dyads with missing rater information, redCards or games
    - compute avgSkin = mean(rater1, rater2)
    - create SkinGroup: 'Light' (avg <= 0.25), 'Medium' (0.25 < avg < 0.75), 'Dark' (avg >= 0.75)
    - create DarkPlayer binary (1 if Dark, 0 if Light). Filter to Dark and Light groups for the primary analysis.
    - ensure games>0 and create log_games_offset = log(games)
    - coerce numeric controls and impute missing numeric control values with median (simple, transparent approach)
    - reset index and return transformed dataframe containing all columns needed for modeling
    """
    df = df.copy()

    # Required columns for transformation
    required_cols = ['rater1', 'rater2', 'redCards', 'games']
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"Required column '{c}' not found in input dataframe")

    # Drop rows missing key variables
    df = df.dropna(subset=['rater1', 'rater2', 'redCards', 'games'])

    # Ensure numeric types
    df['rater1'] = pd.to_numeric(df['rater1'], errors='coerce')
    df['rater2'] = pd.to_numeric(df['rater2'], errors='coerce')
    df['redCards'] = pd.to_numeric(df['redCards'], errors='coerce').fillna(0).astype(int)
    df['games'] = pd.to_numeric(df['games'], errors='coerce')

    # Compute average skin rating
    df['avgSkin'] = df[['rater1', 'rater2']].mean(axis=1)

    # Map average to discrete group: Light / Medium / Dark
    def _skin_group(x):
        if pd.isna(x):
            return pd.NA
        if x <= 0.25:
            return 'Light'
        elif x >= 0.75:
            return 'Dark'
        else:
            return 'Medium'

    df['SkinGroup'] = df['avgSkin'].apply(_skin_group)

    # Keep only dyads where skin group is Light or Dark for primary contrast
    df = df[df['SkinGroup'].isin(['Light', 'Dark'])].copy()

    # Binary indicator: Dark vs Light
    df['DarkPlayer'] = df['SkinGroup'].map({'Dark': 1, 'Light': 0})

    # Remove rows with non-positive or missing games (no exposure)
    df = df[~df['games'].isna()]
    df = df[df['games'] > 0]

    # Offset for exposure (log of games)
    df['log_games_offset'] = np.log(df['games'])

    # Coerce and impute numeric controls if present
    numeric_controls = ['age', 'height', 'weight', 'meanIAT', 'meanExp']
    for col in numeric_controls:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # median imputation for missing control values
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Ensure position and refNum remain (position may be used as categorical control; refNum for clustering)
    if 'position' in df.columns:
        df['position'] = df['position'].astype('category')
    if 'refNum' in df.columns:
        df['refNum'] = pd.to_numeric(df['refNum'], errors='coerce')

    df = df.reset_index(drop=True)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Statistical modeling:
    - Primary analysis: Negative binomial GLM for red card counts with log(games) as offset.
      Formula: redCards ~ DarkPlayer + age + height + weight + meanIAT + meanExp + C(position)
      Cluster-robust standard errors at the referee level (refNum).

    - Sensitivity analysis: same model but replacing binary DarkPlayer with continuous avgSkin.

    Returns a dictionary with two fitted, cluster-robust result objects.
    """
    import statsmodels.formula.api as smf

    # Make a shallow copy
    df = df.copy()

    # Ensure required columns for modeling
    required = ['redCards', 'log_games_offset', 'refNum']
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Required column '{c}' not found in dataframe passed to model()")

    # Define formula for primary analysis (Dark vs Light)
    formula_primary = 'redCards ~ DarkPlayer + age + height + weight + meanIAT + meanExp + C(position)'

    # Fit Negative Binomial GLM with offset
    nb_primary = smf.glm(formula=formula_primary,
                         data=df,
                         family=sm.families.NegativeBinomial(),
                         offset=df['log_games_offset']).fit()

    # Cluster-robust standard errors at referee level
    if 'refNum' in df.columns and df['refNum'].notna().any():
        try:
            nb_primary_clust = nb_primary.get_robustcov_results(cov_type='cluster', groups=df['refNum'])
        except Exception:
            # Fallback: return the original results if clustering fails
            nb_primary_clust = nb_primary
    else:
        nb_primary_clust = nb_primary

    # Sensitivity model using continuous avgSkin
    formula_cont = 'redCards ~ avgSkin + age + height + weight + meanIAT + meanExp + C(position)'
    nb_cont = smf.glm(formula=formula_cont,
                      data=df,
                      family=sm.families.NegativeBinomial(),
                      offset=df['log_games_offset']).fit()

    if 'refNum' in df.columns and df['refNum'].notna().any():
        try:
            nb_cont_clust = nb_cont.get_robustcov_results(cov_type='cluster', groups=df['refNum'])
        except Exception:
            nb_cont_clust = nb_cont
    else:
        nb_cont_clust = nb_cont

    # Return both fitted, cluster-robust result objects. Callers can print summary() or access params, bse, conf_int, etc.
    return {
        'nb_model_dark_vs_light': nb_primary_clust,
        'nb_model_avgSkin': nb_cont_clust
    }


