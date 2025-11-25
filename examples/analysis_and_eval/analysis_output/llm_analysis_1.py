from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/.venv/lib/python3.10/site-packages/blade_bench/datasets/hurricane/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataframe into a form suitable for regression.
    Produces the following final columns used in modeling:
      - Fem_Z: z-scored masfem (higher = more feminine)
      - GenderBinary: original gender_mf (0 male, 1 female)
      - LogDamage: log(ndam15 + 1)
      - LogDeaths: log(alldeaths + 1)  (kept for robustness checks)
      - Wind: wind
      - Category: category
      - MinPressure: min (renamed)
      - Year_Centered: year centered around its mean
      - ElapsedYears: elapsedyrs
      - Source: source (kept as-is for categorical control)

    Notes: we drop observations missing the primary IV (masfem) or primary DV (ndam15),
    coerce numeric columns to numeric types, and create z-scores for the femininity index.
    """
    df = df.copy()

    # Ensure numeric columns are numeric
    num_cols = ['masfem', 'masfem_mturk', 'gender_mf', 'ndam15', 'ndam', 'alldeaths', 'wind', 'category', 'min', 'year', 'elapsedyrs']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing the main independent variable or main dependent variable
    # (masfem for femininity; ndam15 for damage outcome)
    df = df.dropna(subset=['masfem', 'ndam15'])

    # Create outcome: log-transformed 2015-adjusted damage
    df['LogDamage'] = np.log(df['ndam15'].fillna(0) + 1)

    # Secondary outcome for robustness: log-transformed total deaths
    df['alldeaths'] = df.get('alldeaths', pd.Series(index=df.index, dtype=float)).fillna(0)
    df['LogDeaths'] = np.log(df['alldeaths'] + 1)

    # Create z-scored femininity index from masfem
    df['Fem_Z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1.0)

    # Binary gender label (0 male, 1 female)
    if 'gender_mf' in df.columns:
        df['GenderBinary'] = df['gender_mf'].astype(float)
    else:
        df['GenderBinary'] = np.nan

    # Rename / copy key physical control variables to consistent column names used in modeling
    df['Wind'] = df['wind']
    df['Category'] = df['category']
    df['MinPressure'] = df['min']

    # Year centered to aid interpretation
    df['Year_Centered'] = df['year'] - df['year'].mean()

    # Elapsed years (keep as provided)
    df['ElapsedYears'] = df['elapsedyrs']

    # Source: keep as-is for categorical control; ensure no missing values become 'unknown'
    if 'source' in df.columns:
        df['Source'] = df['source'].fillna('unknown').astype(str)
    else:
        df['Source'] = 'unknown'

    # Select and return only the columns necessary for the analysis to make downstream modeling explicit
    keep_cols = ['Fem_Z', 'masfem', 'masfem_mturk', 'GenderBinary', 'LogDamage', 'LogDeaths',
                 'Wind', 'Category', 'MinPressure', 'Year_Centered', 'ElapsedYears', 'Source', 'name', 'year', 'ind']

    # Some of these columns might not exist in every dataset variant; keep those that do
    keep_cols = [c for c in keep_cols if c in df.columns]

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit OLS models to test whether feminine hurricane names predict lower precautionary behavior
    proxies (here: more damage). Main model predicts LogDamage from Fem_Z controlling for
    storm intensity and temporal/source covariates. Robustness model uses LogDeaths as outcome.

    Returns a dictionary with fitted results and prints summaries.
    Uses heteroskedasticity-robust standard errors (HC3).
    """
    import statsmodels.formula.api as smf
    results = {}

    # Ensure the expected columns are present
    required = ['LogDamage', 'Fem_Z', 'Wind', 'Category', 'MinPressure', 'Year_Centered', 'ElapsedYears', 'Source']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Build formula for main model: LogDamage ~ Fem_Z + physical controls + temporal controls + source fixed effects
    formula_main = 'LogDamage ~ Fem_Z + Wind + Category + MinPressure + Year_Centered + ElapsedYears + C(Source)'
    mod_main = smf.ols(formula_main, data=df)
    res_main = mod_main.fit(cov_type='HC3')
    results['main_model'] = res_main

    # Robustness model: deaths
    if 'LogDeaths' in df.columns:
        formula_deaths = 'LogDeaths ~ Fem_Z + Wind + Category + MinPressure + Year_Centered + ElapsedYears + C(Source)'
        mod_deaths = smf.ols(formula_deaths, data=df)
        res_deaths = mod_deaths.fit(cov_type='HC3')
        results['deaths_model'] = res_deaths

    # Additional robustness: use MTurk-rated femininity if available
    if 'masfem_mturk' in df.columns:
        # z-score the mturk measure within the dataframe context
        df = df.copy()
        df['FemMTurk_Z'] = (df['masfem_mturk'] - df['masfem_mturk'].mean()) / (df['masfem_mturk'].std(ddof=0) if df['masfem_mturk'].std(ddof=0) != 0 else 1.0)
        formula_mturk = 'LogDamage ~ FemMTurk_Z + Wind + Category + MinPressure + Year_Centered + ElapsedYears + C(Source)'
        mod_mturk = smf.ols(formula_mturk, data=df)
        res_mturk = mod_mturk.fit(cov_type='HC3')
        results['mturk_model'] = res_mturk

    # Print concise summaries
    print('MAIN MODEL: LogDamage ~ Fem_Z + controls')
    print(results['main_model'].summary())

    if 'deaths_model' in results:
        print('\nROBUSTNESS: LogDeaths ~ Fem_Z + controls')
        print(results['deaths_model'].summary())

    if 'mturk_model' in results:
        print('\nROBUSTNESS: LogDamage ~ masfem_mturk (z-scored) + controls')
        print(results['mturk_model'].summary())

    return results


