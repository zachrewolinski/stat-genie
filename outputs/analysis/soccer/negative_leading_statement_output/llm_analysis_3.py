from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/soccer/negative_leading_statement_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy to avoid modifying original
    df = df.copy()

    # Ensure required numeric columns exist and drop rows missing key variables
    # We need rater1, rater2 for skin tone; redCards and games for the outcome/exposure
    df = df.dropna(subset=['rater1', 'rater2', 'redCards', 'games'])

    # Convert birthday to datetime and compute age at season midpoint (2013-01-01)
    df['birthday'] = pd.to_datetime(df['birthday'], dayfirst=True, errors='coerce', format='%d.%m.%Y')
    reference_date = pd.to_datetime('2013-01-01')
    df['Age'] = (reference_date - df['birthday']).dt.days / 365.25

    # Create average skin-tone rating (continuous)
    df['SkinToneAvg'] = df[['rater1', 'rater2']].mean(axis=1)

    # Create tercile thresholds to define extreme light vs extreme dark
    q_low = df['SkinToneAvg'].quantile(0.33)
    q_high = df['SkinToneAvg'].quantile(0.67)

    # DarkBinary: 1 = dark (>= top tercile), 0 = light (<= bottom tercile), NaN otherwise
    df['DarkBinary'] = np.where(df['SkinToneAvg'] >= q_high, 1,
                                np.where(df['SkinToneAvg'] <= q_low, 0, np.nan))

    # KeepExtreme flag to identify rows appropriate for the binary extreme comparison
    df['KeepExtremes'] = df['DarkBinary'].notnull()

    # Create an offset for exposure (log of number of games in dyad)
    # games should be >= 1 per schema; guard against zeros just in case
    df['games'] = pd.to_numeric(df['games'], errors='coerce').fillna(0).astype(int)
    df = df[df['games'] > 0]  # keep dyads with at least one game
    df['log_games'] = np.log(df['games'])

    # Create dummies for position and leagueCountry (drop first level to avoid multicollinearity)
    # Use stable prefixes used later in modeling: pos_ and leagueCountry_
    pos_dummies = pd.get_dummies(df['position'].fillna('Unknown'), prefix='pos', drop_first=True)
    league_dummies = pd.get_dummies(df['leagueCountry'].fillna('Unknown'), prefix='leagueCountry', drop_first=True)

    # Concatenate the dummy columns onto the dataframe
    df = pd.concat([df, pos_dummies, league_dummies], axis=1)

    # Ensure meanIAT and meanExp are numeric
    df['meanIAT'] = pd.to_numeric(df['meanIAT'], errors='coerce')
    df['meanExp'] = pd.to_numeric(df['meanExp'], errors='coerce')

    # Keep only columns needed for modeling plus identifiers
    # (We keep many columns because model function will select the pos_/leagueCountry_ columns programmatically.)
    # Return the transformed DataFrame
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fits two negative-binomial GLMs for redCards (counts) using games as exposure (offset):
      1) Continuous model: SkinToneAvg as a continuous predictor (full sample).
      2) Binary extremes model: DarkBinary (dark vs light extremes) using only top/bottom terciles.

    Returns a dict with the fitted (cluster-robust) result objects for both models.
    """
    results = {}

    # Build base controls list
    base_controls = ['Age', 'height', 'weight', 'meanIAT', 'meanExp']

    # Identify position and league dummy columns created by transform function
    pos_cols = [c for c in df.columns if c.startswith('pos_')]
    league_cols = [c for c in df.columns if c.startswith('leagueCountry_')]

    # Ensure we have at least one position/league column (if not, continue without them)
    control_cols = [c for c in base_controls if c in df.columns] + pos_cols + league_cols

    # Model 1: continuous SkinToneAvg on full sample
    # Prepare exogenous matrix
    exog_cont_cols = ['SkinToneAvg'] + control_cols
    # Drop rows with missing exogenous values
    df_cont = df.dropna(subset=['redCards', 'log_games'] + exog_cont_cols)
    X_cont = df_cont[exog_cont_cols]
    X_cont = sm.add_constant(X_cont, has_constant='add')
    y_cont = df_cont['redCards']

    # Fit negative binomial GLM with offset = log_games
    try:
        model_cont = sm.GLM(y_cont, X_cont, family=sm.families.NegativeBinomial(), offset=df_cont['log_games']).fit()
        # Cluster robust SEs by referee (refNum)
        if 'refNum' in df_cont.columns:
            model_cont_clust = model_cont.get_robustcov_results(cov_type='cluster', groups=df_cont['refNum'])
        else:
            model_cont_clust = model_cont.get_robustcov_results(cov_type='HC3')
        results['model_continuous'] = model_cont_clust
    except Exception as e:
        results['model_continuous'] = e

    # Model 2: binary dark vs light using extreme terciles only
    # Keep only rows marked as extremes in transform (KeepExtremes == True)
    if 'KeepExtremes' in df.columns:
        df_bin = df[df['KeepExtremes'] == True].copy()
    else:
        # Fallback: keep rows where DarkBinary is not null
        df_bin = df[df['DarkBinary'].notnull()].copy()

    exog_bin_cols = ['DarkBinary'] + control_cols
    # Drop rows with missing values for the binary model
    df_bin = df_bin.dropna(subset=['redCards', 'log_games'] + exog_bin_cols)

    if df_bin.shape[0] > 0:
        X_bin = df_bin[exog_bin_cols]
        X_bin = sm.add_constant(X_bin, has_constant='add')
        y_bin = df_bin['redCards']
        try:
            model_bin = sm.GLM(y_bin, X_bin, family=sm.families.NegativeBinomial(), offset=df_bin['log_games']).fit()
            if 'refNum' in df_bin.columns:
                model_bin_clust = model_bin.get_robustcov_results(cov_type='cluster', groups=df_bin['refNum'])
            else:
                model_bin_clust = model_bin.get_robustcov_results(cov_type='HC3')
            results['model_binary_extremes'] = model_bin_clust
        except Exception as e:
            results['model_binary_extremes'] = e
    else:
        results['model_binary_extremes'] = ValueError('No rows available after filtering to extremes for binary comparison.')

    # Return results (each entry is a fitted Results object or an Exception)
    return results


