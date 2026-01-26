from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/noperturb_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataframe for modeling.

    Produces (keeps) the following columns used in the models:
      - masfem (original continuous femininity score)
      - masfem_z (z-scored femininity)
      - gender_mf (binary female name indicator)
      - alldeaths (raw death count)
      - ndam15 (damage in 2015 dollars)
      - log_alldeaths (log(1 + alldeaths))
      - log_ndam15 (log(1 + ndam15))
      - wind, min, category, year, elapsedyrs, source
      - severity_idx (standardized index combining wind, -min, and category)

    The function avoids aggressive row-dropping so users can choose the appropriate subset for each analysis.
    """
    df = df.copy()

    # Ensure numeric columns are numeric (coerce invalid -> NaN)
    numeric_cols = ['masfem', 'masfem_mturk', 'wind', 'min', 'category', 'alldeaths', 'ndam15', 'ndam', 'year', 'elapsedyrs', 'gender_mf']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Create logged outcome variables (log(1 + x)) to handle zeros and skew
    if 'alldeaths' in df.columns:
        df['log_alldeaths'] = np.log1p(df['alldeaths'])
    else:
        df['log_alldeaths'] = np.nan

    if 'ndam15' in df.columns:
        df['log_ndam15'] = np.log1p(df['ndam15'])
    else:
        df['log_ndam15'] = np.nan

    # Standardize the masfem score for interpretability (z-score)
    if 'masfem' in df.columns:
        mas_mean = df['masfem'].mean(skipna=True)
        mas_sd = df['masfem'].std(skipna=True)
        # Avoid division by zero
        if pd.isna(mas_sd) or mas_sd == 0:
            df['masfem_z'] = df['masfem'] - mas_mean
        else:
            df['masfem_z'] = (df['masfem'] - mas_mean) / mas_sd
    else:
        df['masfem_z'] = np.nan

    # Build a severity index from wind, inverse pressure (lower pressure = worse), and category
    # Compute z-scores column-wise, handling missing data
    def zscore(series):
        return (series - series.mean(skipna=True)) / series.std(skipna=True)

    # Only compute if constituent columns exist
    has_wind = 'wind' in df.columns
    has_min = 'min' in df.columns
    has_cat = 'category' in df.columns

    # Create temporary z columns
    if has_wind:
        df['z_wind'] = zscore(df['wind'])
    else:
        df['z_wind'] = np.nan

    if has_min:
        # lower pressure -> more severe, so negate before z-scoring
        df['z_negmin'] = zscore(-df['min'])
    else:
        df['z_negmin'] = np.nan

    if has_cat:
        df['z_category'] = zscore(df['category'])
    else:
        df['z_category'] = np.nan

    # Combine into severity_idx as the mean of available z-components (so missing one won't drop the index entirely)
    df['severity_idx'] = df[['z_wind', 'z_negmin', 'z_category']].mean(axis=1, skipna=True)

    # Keep core columns used in modeling so the returned DF is explicit about what's available
    keep_cols = [
        'masfem', 'masfem_z', 'gender_mf', 'alldeaths', 'log_alldeaths', 'ndam15', 'log_ndam15',
        'wind', 'min', 'category', 'severity_idx', 'year', 'elapsedyrs', 'source'
    ]

    # Add any of these columns that don't exist (as NaN) so the final df has consistent columns
    for c in keep_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Return the dataframe with all other columns preserved but ensure the modeling columns exist
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Runs statistical models testing whether more feminine hurricane names predict greater harm
    (used as a proxy for fewer precautionary measures).

    Models returned:
      - ols_deaths: OLS on log_alldeaths with robust (HC3) SEs
      - ols_damage: OLS on log_ndam15 with robust (HC3) SEs
      - nb_deaths: Negative binomial regression on raw alldeaths (robustness check)

    Each model uses the same set of regressors: masfem_z, gender_mf, severity_idx, year, elapsedyrs,
    and categorical dummies for source (drop_first=True).

    The function returns a dict with fitted result objects. Users can inspect .summary() on each.
    """
    # Local imports for discrete models
    from statsmodels.discrete.discrete_model import NegativeBinomial

    # Work on a copy to avoid modifying caller's DF
    df = df.copy()

    # Select rows with the core predictor and at least one outcome present - we'll build separate subsets
    # Build the base regressor matrix
    base_regs = ['masfem_z', 'gender_mf', 'severity_idx', 'year', 'elapsedyrs']
    for col in base_regs:
        if col not in df.columns:
            df[col] = np.nan

    # Build source dummies (drop first to avoid full collinearity)
    if 'source' in df.columns:
        source_dummies = pd.get_dummies(df['source'].astype(str), prefix='source', drop_first=True)
    else:
        source_dummies = pd.DataFrame(index=df.index)

    # Helper to prepare X (ensures same rows as y by joining and dropping NA rows)
    def prepare_X_y(df_local, y_col):
        # y_col is expected to be already in df_local (raw or log)
        subset = df_local[base_regs + [y_col]].join(source_dummies)
        subset = subset.dropna(subset=base_regs + [y_col])
        y = subset[y_col]
        X = subset.drop(columns=[y_col])
        X = sm.add_constant(X, has_constant='add')
        return X, y

    results = {}

    # 1) OLS on log_alldeaths
    if 'log_alldeaths' in df.columns:
        X_deaths, y_deaths = prepare_X_y(df, 'log_alldeaths')
        if len(y_deaths) > 0:
            ols_deaths = sm.OLS(y_deaths, X_deaths).fit(cov_type='HC3')
            results['ols_deaths'] = ols_deaths
        else:
            results['ols_deaths'] = None
    else:
        results['ols_deaths'] = None

    # 2) OLS on log_ndam15
    if 'log_ndam15' in df.columns:
        X_damage, y_damage = prepare_X_y(df, 'log_ndam15')
        if len(y_damage) > 0:
            ols_damage = sm.OLS(y_damage, X_damage).fit(cov_type='HC3')
            results['ols_damage'] = ols_damage
        else:
            results['ols_damage'] = None
    else:
        results['ols_damage'] = None

    # 3) Negative Binomial on raw alldeaths as a robustness check (count model)
    # Use same regressors as the deaths OLS. Fit only on rows where alldeaths and regressors are present.
    if 'alldeaths' in df.columns:
        X_nb, y_nb = prepare_X_y(df, 'alldeaths')
        # prepare_X_y expects y_col to be present; it will cast any numeric column accordingly
        # statsmodels expects endog to be non-negative integers for NB; coerce/floor if necessary
        # We'll drop negative values if any (shouldn't be) and ensure integer type
        valid_ix = (y_nb >= 0)
        y_nb = y_nb[valid_ix]
        X_nb = X_nb.loc[valid_ix.index][valid_ix]

        if len(y_nb) > 0:
            try:
                nb_model = NegativeBinomial(y_nb, X_nb).fit(disp=False)
                results['nb_deaths'] = nb_model
            except Exception as e:
                # If NB fails to converge, return the exception for debugging
                results['nb_deaths'] = e
        else:
            results['nb_deaths'] = None
    else:
        results['nb_deaths'] = None

    return results


