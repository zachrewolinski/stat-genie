from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/soccer/anonymize_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset into an analysis-ready dataframe.

    Produces the following key columns used in modeling (exact names used later):
      - RedCards: count of red cards (from feature16)
      - Matches: number of matches in dyad (from feature9)
      - SkinTone: continuous average of two raters (feature18, feature19)
      - DarkBinary: binary 1=Dark (SkinTone >= 0.75), 0=Light (SkinTone <= 0.25), NaN otherwise
      - Position, LeagueCountry, RefereeID, RefCountry_IAT, RefCountry_Explicit, Goals, YellowCards, HeightCM, WeightKG, Age

    Notes:
      - Birthdate (feature5) parsed assuming dd.mm.yyyy; Age approximated as 2013 - birth_year.
      - The dataset contains normalized rater scores on a 0-1 scale (5 levels). We average across raters if both or one is present.
    """
    df = df.copy()

    # --- Rename columns to clear variable names ---
    rename_map = {
        'feature9': 'Matches',            # number of games in dyad
        'feature16': 'RedCards',          # number of red cards
        'feature18': 'Rater1',            # skin rating rater 1 (0-1)
        'feature19': 'Rater2',            # skin rating rater 2 (0-1)
        'feature8': 'Position',
        'feature4': 'LeagueCountry',
        'feature20': 'RefereeID',
        'feature21': 'RefCountryID',
        'feature22': 'RefCountry_IAT',
        'feature25': 'RefCountry_Explicit',
        'feature13': 'Goals',
        'feature14': 'YellowCards',
        'feature6': 'HeightCM',
        'feature7': 'WeightKG',
        'feature5': 'Birthdate'
    }
    df = df.rename(columns=rename_map)

    # Ensure numeric columns are numeric
    numeric_cols = ['Matches', 'RedCards', 'Rater1', 'Rater2', 'RefCountry_IAT', 'RefCountry_Explicit',
                    'Goals', 'YellowCards', 'HeightCM', 'WeightKG']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Compute continuous average skin tone (use available rater(s))
    df['SkinTone'] = df[['Rater1', 'Rater2']].mean(axis=1, skipna=True)

    # Create binary dark vs light indicator for the focal comparison.
    # Mapping uses the normalized 5-point levels: 0.0,0.25,0.5,0.75,1.0
    # Dark if avg >= 0.75, Light if avg <= 0.25, otherwise set NaN (ambiguous/mid)
    df['DarkBinary'] = np.where(df['SkinTone'] >= 0.75, 1,
                                np.where(df['SkinTone'] <= 0.25, 0, np.nan))

    # Parse birthdate and compute approximate age at season midpoint (2013)
    if 'Birthdate' in df.columns:
        # Expecting format dd.mm.yyyy (e.g., 06.04.1989)
        df['Birthdate_parsed'] = pd.to_datetime(df['Birthdate'], dayfirst=True, errors='coerce')
        df['Age'] = np.where(df['Birthdate_parsed'].notna(), 2013 - df['Birthdate_parsed'].dt.year, np.nan)
    else:
        df['Age'] = np.nan

    # Safety: ensure Matches positive; offset requires Matches > 0. Filtering of zero-match rows can be done before modeling.
    df['Matches'] = pd.to_numeric(df['Matches'], errors='coerce')

    # Keep the relevant final columns (but return whole df copy so user can inspect raw columns too)
    # Ensure categorical columns are strings (so patsy/statsmodels handle them as categorical when using C(...))
    for c in ['Position', 'LeagueCountry']:
        if c in df.columns:
            df[c] = df[c].astype('category')

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Run count models to test whether darker-skinned players receive more red cards.

    Approach:
      - Primary analysis: Negative binomial regression of RedCards with binary DarkBinary predictor
        (Dark vs Light). Uses log(Matches) as an offset (exposure) and clusters SEs by RefereeID.
        Only dyads classified as Dark or Light (no ambiguous/mid) are included.
      - Secondary analysis: Same model but using continuous SkinTone in the full sample.

    Returns a dictionary with fitted model results (both raw and clustered covariance versions).
    """
    import statsmodels.api as sm

    results = {}

    # Ensure necessary columns exist
    required = ['RedCards', 'Matches', 'RefereeID']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Required column {c} not found in dataframe")

    # --- Primary: binary dark vs light (exclude ambiguous) ---
    df_primary = df.dropna(subset=['DarkBinary', 'RedCards', 'Matches']).copy()
    df_primary = df_primary[df_primary['Matches'] > 0].copy()
    if df_primary.shape[0] == 0:
        raise ValueError('No rows available for primary (binary) analysis after filtering')

    # Create offset
    df_primary['Offset'] = np.log(df_primary['Matches'])

    # Formula: count of red cards with exposure offset. Position and LeagueCountry as categorical fixed effects.
    formula_bin = (
        'RedCards ~ DarkBinary + C(Position) + C(LeagueCountry) + Age + Goals + YellowCards'
        ' + HeightCM + WeightKG + RefCountry_IAT + RefCountry_Explicit'
    )

    # Fit negative binomial via GLM (handles overdispersion vs Poisson).
    model_nb_bin = sm.GLM.from_formula(formula_bin, data=df_primary,
                                      family=sm.families.NegativeBinomial(),
                                      offset=df_primary['Offset'])
    res_nb_bin = model_nb_bin.fit()

    # Clustered standard errors by RefereeID to account for within-referee dependence
    try:
        res_nb_bin_clustered = res_nb_bin.get_robustcov_results(cov_type='cluster',
                                                                groups=df_primary['RefereeID'])
    except Exception:
        # Fallback in case clustering fails (return raw results)
        res_nb_bin_clustered = res_nb_bin

    results['nb_binary_raw'] = res_nb_bin
    results['nb_binary_clustered'] = res_nb_bin_clustered

    # --- Secondary: continuous skin tone (use full sample where SkinTone available) ---
    df_cont = df.dropna(subset=['SkinTone', 'RedCards', 'Matches']).copy()
    df_cont = df_cont[df_cont['Matches'] > 0].copy()
    if df_cont.shape[0] == 0:
        raise ValueError('No rows available for continuous SkinTone analysis after filtering')

    df_cont['Offset'] = np.log(df_cont['Matches'])

    formula_cont = (
        'RedCards ~ SkinTone + C(Position) + C(LeagueCountry) + Age + Goals + YellowCards'
        ' + HeightCM + WeightKG + RefCountry_IAT + RefCountry_Explicit'
    )

    model_nb_cont = sm.GLM.from_formula(formula_cont, data=df_cont,
                                       family=sm.families.NegativeBinomial(),
                                       offset=df_cont['Offset'])
    res_nb_cont = model_nb_cont.fit()

    try:
        res_nb_cont_clustered = res_nb_cont.get_robustcov_results(cov_type='cluster',
                                                                  groups=df_cont['RefereeID'])
    except Exception:
        res_nb_cont_clustered = res_nb_cont

    results['nb_continuous_raw'] = res_nb_cont
    results['nb_continuous_clustered'] = res_nb_cont_clustered

    # --- Optional diagnostics: compare Poisson dispersion on primary sample ---
    try:
        model_pois = sm.GLM.from_formula(formula_bin, data=df_primary,
                                         family=sm.families.Poisson(),
                                         offset=df_primary['Offset'])
        res_pois = model_pois.fit()
        pearson_chi2 = (res_pois.resid_pearson ** 2).sum()
        dispersion = pearson_chi2 / res_pois.df_resid if res_pois.df_resid > 0 else np.nan
        results['poisson_dispersion_primary'] = dispersion
    except Exception:
        results['poisson_dispersion_primary'] = None

    return results


