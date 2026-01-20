from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/hurricane/replace_with_rvs_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataframe into a modeling-ready dataframe.
    Adds/renames columns used in the statistical models below.

    Produced columns (exact names used later):
      - Deaths: integer count from alldeaths
      - LogDeaths: log1p(Deaths) (kept for diagnostics/robustness)
      - Damage: ndam15 (inflation/adj damage)
      - LogDamage: log1p(Damage)
      - FemaleName: binary indicator equal to gender_mf (0/1)
      - NameFemScore: original masfem score
      - NameFem_z: standardized masfem (z-score)
      - MaxWind: wind speed
      - MinPressure: min (pressure)
      - Category: integer category
      - YearCentered: year - year.mean()
      - ElapsedYrs: elapsedyrs
      - Source: source (string/category)
    """
    df = df.copy()

    # Normalize column names if necessary (use the names specified in dataset schema)
    # Ensure numeric conversions where appropriate
    num_cols = ['alldeaths', 'ndam15', 'masfem', 'masfem_mturk', 'gender_mf', 'wind', 'min', 'category', 'year', 'elapsedyrs']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Keep source as string (may be categorical)
    if 'source' in df.columns:
        df['source'] = df['source'].astype('str').fillna('unknown')

    # Drop rows missing essential fields for modeling
    required = ['alldeaths', 'masfem', 'gender_mf', 'wind', 'min', 'category', 'year', 'ndam15']
    present_required = [c for c in required if c in df.columns]
    df = df.dropna(subset=present_required).reset_index(drop=True)

    # Dependent variables
    df['Deaths'] = df['alldeaths'].astype(int)
    df['LogDeaths'] = np.log1p(df['Deaths'].clip(lower=0))

    # Damage measures
    df['Damage'] = df['ndam15']
    df['LogDamage'] = np.log1p(df['Damage'].clip(lower=0))

    # Independent variables
    # Binary female name indicator (0 = male, 1 = female)
    df['FemaleName'] = df['gender_mf'].astype(int)

    # Continuous femininity score and standardized version
    df['NameFemScore'] = df['masfem'].astype(float)
    # Use population std (ddof=0) to be explicit
    mean_score = df['NameFemScore'].mean()
    std_score = df['NameFemScore'].std(ddof=0)
    if std_score == 0 or np.isnan(std_score):
        df['NameFem_z'] = 0.0
    else:
        df['NameFem_z'] = (df['NameFemScore'] - mean_score) / std_score

    # Physical controls
    df['MaxWind'] = df['wind'].astype(float)
    df['MinPressure'] = df['min'].astype(float)
    df['Category'] = df['category'].astype(int)

    # Time controls
    df['YearCentered'] = df['year'].astype(int) - df['year'].astype(int).mean()
    df['ElapsedYrs'] = df['elapsedyrs'].astype(float)

    # Source (keep original string/category)
    df['Source'] = df['source'].astype(str)

    # Return only the columns needed for modeling + a few diagnostics (but keep full df copy)
    # We'll return the full df with added columns so consumer can choose.
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Runs statistical models testing whether feminine-named hurricanes lead to different outcomes
    (used here as a proxy for fewer precautionary actions by the public).

    Models returned:
      - nb_model_female_binary: Negative Binomial regression for Deaths with FemaleName (binary)
      - nb_model_femscore: Negative Binomial regression for Deaths with NameFem_z (continuous)
      - ols_logdamage: OLS regression for LogDamage with NameFem_z (robust HC3 SEs)

    Returns a dict of fitted statsmodels results objects.
    """
    df = df.copy()

    # Verify required columns
    required_model_cols = ['Deaths', 'FemaleName', 'NameFem_z', 'MaxWind', 'MinPressure', 'Category', 'YearCentered', 'ElapsedYrs', 'Source', 'LogDamage']
    missing = [c for c in required_model_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for model: {missing}")

    # Base covariates (physical intensity + time + elapsedyrs)
    base_covs = ['MaxWind', 'MinPressure', 'Category', 'YearCentered', 'ElapsedYrs']
    X_base = df[base_covs].astype(float).copy()

    # One-hot encode Source, drop first to avoid multicollinearity
    source_dummies = pd.get_dummies(df['Source'].fillna('unknown'), prefix='Source', drop_first=True)
    X_base = pd.concat([X_base, source_dummies.reset_index(drop=True)], axis=1)

    # 1) Negative Binomial with binary FemaleName
    X_nb_bin = X_base.copy()
    X_nb_bin['FemaleName'] = df['FemaleName'].astype(int).values
    X_nb_bin = sm.add_constant(X_nb_bin, has_constant='add')
    y_deaths = df['Deaths'].astype(int)

    # Fit Negative Binomial (GLM) for counts; default link is log
    try:
        nb_model_female_binary = sm.GLM(y_deaths, X_nb_bin, family=sm.families.NegativeBinomial()).fit()
    except Exception:
        # Fallback to Poisson if NegativeBinomial fails to converge
        nb_model_female_binary = sm.GLM(y_deaths, X_nb_bin, family=sm.families.Poisson()).fit()

    # 2) Negative Binomial with continuous standardized femininity score
    X_nb_cont = X_base.copy()
    X_nb_cont['NameFem_z'] = df['NameFem_z'].astype(float).values
    X_nb_cont = sm.add_constant(X_nb_cont, has_constant='add')
    try:
        nb_model_femscore = sm.GLM(y_deaths, X_nb_cont, family=sm.families.NegativeBinomial()).fit()
    except Exception:
        nb_model_femscore = sm.GLM(y_deaths, X_nb_cont, family=sm.families.Poisson()).fit()

    # 3) OLS on log damage as a robustness check (continuous femininity measure)
    X_ols = X_nb_cont.copy()  # includes const and NameFem_z
    y_logdamage = df['LogDamage'].astype(float)
    ols_model = sm.OLS(y_logdamage, X_ols).fit(cov_type='HC3')

    # Return the fitted results objects for downstream inspection
    results = {
        'nb_model_female_binary': nb_model_female_binary,
        'nb_model_femscore': nb_model_femscore,
        'ols_logdamage': ols_model
    }
    return results


