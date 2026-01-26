from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/soccer/shuffle_names_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset to the analysis-ready dataframe.

    Steps:
    - Keep rows with photos and valid rater scores and with non-missing redCards and games (exposure).
    - Compute rater_mean from rater1 and rater2.
    - Restrict to players with photos (photoID > 0) to ensure skin tone coding exists.
    - Define skin-tone tertiles on rater_mean and keep only the bottom tertile (Light) and top tertile (Dark) to form a clear contrast.
    - Create binary DarkSkin indicator: 1 for Dark (top tertile), 0 for Light (bottom tertile).
    - Create exposure_games from 'games' and ensure it's positive. Convert redCards to integer.
    - Standardize continuous controls: playerShort (proxy for prior yellow cards), meanIAT (implicit bias), club (explicit bias).

    Returns the dataframe with columns used in the model: ['redCards', 'exposure_games', 'DarkSkin', 'rater_mean', 'playerShort_z', 'meanIAT_z', 'club_z']
    """
    df = df.copy()

    # Keep needed raw columns — tolerant to original column types
    required_cols = ['rater1', 'rater2', 'photoID', 'redCards', 'games', 'playerShort', 'meanIAT', 'club']
    for c in required_cols:
        if c not in df.columns:
            # If a required column is missing, raise an informative error so user can inspect
            raise KeyError(f"Required column '{c}' not found in input dataframe.")

    # Convert to numeric where appropriate
    df['rater1'] = pd.to_numeric(df['rater1'], errors='coerce')
    df['rater2'] = pd.to_numeric(df['rater2'], errors='coerce')
    df['photoID'] = pd.to_numeric(df['photoID'], errors='coerce')
    df['redCards'] = pd.to_numeric(df['redCards'], errors='coerce')
    df['games'] = pd.to_numeric(df['games'], errors='coerce')
    df['playerShort'] = pd.to_numeric(df['playerShort'], errors='coerce')
    df['meanIAT'] = pd.to_numeric(df['meanIAT'], errors='coerce')
    df['club'] = pd.to_numeric(df['club'], errors='coerce')

    # Compute rater mean and keep only rows with available rater scores
    df['rater_mean'] = df[['rater1', 'rater2']].mean(axis=1)

    # Keep only rows with photo and non-missing rater_mean, redCards, games
    df = df[(df['photoID'].notna()) & (df['photoID'] > 0) & (df['rater_mean'].notna())]
    df = df[df['redCards'].notna() & df['games'].notna()]

    # Ensure exposure is positive and integer-like (if zero or negative remove)
    df = df[df['games'] > 0].copy()
    df['exposure_games'] = df['games'].astype(float)

    # Cast redCards to integer counts (if fractional due to bad input, take round then clip)
    # Protect against NaN by filling with 0 before rounding if necessary (but we filtered NaNs above)
    df['redCards'] = (df['redCards'].round().astype(int)).clip(lower=0)

    # If after filtering there are no rows, return empty dataframe with expected columns to let caller handle it
    # Create skin tone tertiles and keep only Light (bottom tertile) and Dark (top tertile)
    if df.shape[0] == 0:
        # Create the expected columns so downstream code can at least inspect that transform ran
        df['SkinTone'] = pd.Series(dtype=object)
        df['DarkSkin'] = pd.Series(dtype=int)
        df['playerShort_z'] = pd.Series(dtype=float)
        df['meanIAT_z'] = pd.Series(dtype=float)
        df['club_z'] = pd.Series(dtype=float)
        # Ensure model columns exist
        model_cols = ['redCards', 'exposure_games', 'DarkSkin', 'rater_mean', 'playerShort_z', 'meanIAT_z', 'club_z']
        for col in model_cols:
            if col not in df.columns:
                df[col] = pd.Series(dtype=float)
        return df

    q_low = df['rater_mean'].quantile(1.0/3.0)
    q_high = df['rater_mean'].quantile(2.0/3.0)

    def tone_label(x):
        if x <= q_low:
            return 'Light'
        elif x >= q_high:
            return 'Dark'
        else:
            return 'Medium'

    df['SkinTone'] = df['rater_mean'].apply(tone_label)

    # Keep only light and dark extremes to make a clearer contrast for the tested hypothesis
    df = df[df['SkinTone'].isin(['Light', 'Dark'])].copy()

    # Binary indicator: DarkSkin = 1 for Dark, 0 for Light
    df['DarkSkin'] = (df['SkinTone'] == 'Dark').astype(int)

    # Standardize continuous controls (z-score). Use ddof=0 for population-style standardization; ddof=1 also acceptable.
    for col, out_col in [('playerShort', 'playerShort_z'), ('meanIAT', 'meanIAT_z'), ('club', 'club_z')]:
        # Some columns may have all NA after filtering; handle gracefully
        if df[col].notna().any():
            mean = df[col].mean()
            std = df[col].std(ddof=0)
            if std == 0 or np.isnan(std):
                # If no variation, set standardized value to 0 for all non-missing entries and also for missing later
                df[out_col] = 0.0
            else:
                df[out_col] = (df[col] - mean) / std
            # For any rows where the original value was missing, fill the standardized column with 0 (representing mean)
            df[out_col] = df[out_col].fillna(0.0)
        else:
            # If the entire column is missing for the kept rows, create a column filled with 0.0
            df[out_col] = 0.0

    # Keep only columns needed for modeling (but don't drop others in case user needs them)
    model_cols = ['redCards', 'exposure_games', 'DarkSkin', 'rater_mean', 'playerShort_z', 'meanIAT_z', 'club_z']
    missing_model_cols = [c for c in model_cols if c not in df.columns]
    if missing_model_cols:
        raise KeyError(f"Missing expected columns after transform: {missing_model_cols}")

    # Reorder columns to put model columns first for convenience
    other_cols = [c for c in df.columns if c not in model_cols]
    df = df[model_cols + other_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a negative binomial GLM predicting number of red cards in a player-referee dyad.

    Model specification:
    - Dependent variable: redCards (count)
    - Main predictor: DarkSkin (binary: 1 = Dark, 0 = Light)
    - Controls: meanIAT_z, club_z, playerShort_z, rater_mean (continuous)
    - Exposure (offset): exposure_games (log offset)

    Returns a dictionary with the fitted model object and a small table of incidence rate ratios (IRRs).
    """
    import statsmodels.formula.api as smf

    # Basic checks
    for c in ['redCards', 'exposure_games', 'DarkSkin']:
        if c not in df.columns:
            raise KeyError(f"Column {c} required for modeling not found in dataframe.")

    # Remove any rows with missing values in modeling columns
    model_df = df[['redCards', 'exposure_games', 'DarkSkin', 'rater_mean', 'playerShort_z', 'meanIAT_z', 'club_z']].copy()
    # At this point, transform should have ensured the z-scored controls are non-missing (filled with 0.0 where necessary).
    # But to be robust, replace any remaining NA in control columns with 0.0 (equivalent to mean imputation in standardized space).
    for col in ['rater_mean', 'playerShort_z', 'meanIAT_z', 'club_z']:
        if col in model_df.columns:
            model_df[col] = model_df[col].fillna(0.0)

    model_df = model_df.dropna(subset=['redCards', 'exposure_games', 'DarkSkin'])

    # Check that we have data to fit the model
    if model_df.shape[0] == 0:
        raise ValueError("No observations remain after dropping missing values. Cannot fit model.")

    # Ensure exposures are positive and finite
    if not np.all(np.isfinite(model_df['exposure_games'])) or (model_df['exposure_games'] <= 0).any():
        raise ValueError("All exposure_games must be positive and finite for the offset. Check the transformed data.")

    # Fit negative binomial GLM with log(exposure_games) as offset
    formula = 'redCards ~ DarkSkin + meanIAT_z + club_z + playerShort_z + rater_mean'
    # Use statsmodels' GLM with NegativeBinomial family and offset
    nb_model = smf.glm(formula=formula,
                       data=model_df,
                       family=sm.families.NegativeBinomial(),
                       offset=np.log(model_df['exposure_games']))
    results = nb_model.fit()

    # Compute incidence rate ratios (IRR = exp(coef)) and CIs
    params = results.params
    conf = results.conf_int()
    irr = pd.DataFrame({
        'coef': params,
        'irr': np.exp(params),
        'ci_lower': np.exp(conf[0]),
        'ci_upper': np.exp(conf[1]),
        'pvalue': results.pvalues
    })

    # Arrange IRR table
    irr = irr.loc[params.index]

    # Print brief summary to console (helpful when running interactively)
    print(results.summary())
    print('\nIncidence Rate Ratios (IRR) with 95% CI:')
    print(irr)

    return {'results': results, 'irr_table': irr}