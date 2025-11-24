from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/projects/binyu/hao_huang/stat-genie/.venv/lib/python3.11/site-packages/blade_bench/datasets/hurricane/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataframe to the analysis dataframe.

    Produces the following columns used in modeling:
      - FeminineScore_z: standardized (z) version of 'masfem'
      - FemaleName: binary indicator from 'gender_mf' (0/1)
      - FeminineMTurk_z: standardized (z) version of 'masfem_mturk'
      - LogDeaths: log(1 + alldeaths)
      - LogDamage: log(1 + ndam15)  (kept for supplementary models)
      - MaxWind: copied from 'wind'
      - MinPressure: copied from 'min'
      - Category: copied from 'category'
      - Year: copied from 'year'
      - ElapsedYears: copied from 'elapsedyrs'
      - Source: copied from 'source' (kept as categorical)
    """
    # work on a copy
    df = df.copy()

    # Required columns for analysis
    required_cols = [
        'masfem', 'gender_mf', 'alldeaths', 'ndam15', 'wind', 'min',
        'category', 'year', 'elapsedyrs', 'masfem_mturk', 'source'
    ]

    # Drop rows missing any of the key variables used here
    df = df.dropna(subset=required_cols)

    # Create primary IVs
    df['FeminineScore'] = df['masfem'].astype(float)
    # Standardize femininity score (z-score)
    df['FeminineScore_z'] = (df['FeminineScore'] - df['FeminineScore'].mean()) / df['FeminineScore'].std(ddof=0)

    # Binary female name indicator (ensure integer 0/1)
    # In the dataset gender_mf: 0 = male, 1 = female
    df['FemaleName'] = df['gender_mf'].astype(int)

    # Alternative MTurk femininity rating standardized
    df['FeminineMTurk'] = df['masfem_mturk'].astype(float)
    df['FeminineMTurk_z'] = (df['FeminineMTurk'] - df['FeminineMTurk'].mean()) / df['FeminineMTurk'].std(ddof=0)

    # Dependent variables: transform highly skewed counts/monetary values
    df['alldeaths'] = pd.to_numeric(df['alldeaths'], errors='coerce').fillna(0).astype(float)
    df['LogDeaths'] = np.log1p(df['alldeaths'])

    df['ndam15'] = pd.to_numeric(df['ndam15'], errors='coerce').fillna(0).astype(float)
    df['LogDamage'] = np.log1p(df['ndam15'])

    # Controls: copy and coerce types
    df['MaxWind'] = pd.to_numeric(df['wind'], errors='coerce')
    df['MinPressure'] = pd.to_numeric(df['min'], errors='coerce')
    df['Category'] = pd.to_numeric(df['category'], errors='coerce')
    df['Year'] = pd.to_numeric(df['year'], errors='coerce')
    df['ElapsedYears'] = pd.to_numeric(df['elapsedyrs'], errors='coerce')
    df['Source'] = df['source'].astype('category')

    # Final drop of any rows that became NA after coercion
    final_cols = [
        'FeminineScore_z', 'FemaleName', 'FeminineMTurk_z', 'LogDeaths', 'LogDamage',
        'MaxWind', 'MinPressure', 'Category', 'Year', 'ElapsedYears', 'Source'
    ]
    df = df.dropna(subset=final_cols)

    # Optionally, center Year for better interpretability in models
    df['Year_c'] = df['Year'] - df['Year'].mean()

    # Return only the columns necessary for analysis (plus a few useful originals)
    keep_cols = final_cols + ['Year_c', 'alldeaths', 'ndam15', 'masfem', 'masfem_mturk']
    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Run statistical models that test whether feminine hurricane names predict fewer precautions (proxied by greater fatalities and damage),
    controlling for storm physical severity and time/source controls.

    Returns a dict of fitted model result objects (statsmodels result instances).
    Primary models:
      - OLS on LogDeaths (robust HC3 SE)
      - Negative Binomial on raw alldeaths (counts)
      - OLS on LogDamage (robust HC3 SE) as a damage-based robustness check
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    results = {}

    # Ensure df passed in is the transformed dataframe (contains columns used below)
    # Model 1: OLS on log(1 + deaths) using continuous femininity score
    formula_ols_deaths = (
        'LogDeaths ~ FeminineScore_z + MaxWind + MinPressure + Category + Year_c + ElapsedYears + FeminineMTurk_z + C(Source)'
    )
    ols_deaths = smf.ols(formula_ols_deaths, data=df).fit(cov_type='HC3')
    results['ols_logdeaths'] = ols_deaths

    # Model 2: Negative binomial on raw death counts (alldeaths)
    # Build design matrix manually (statsmodels GLM with NegativeBinomial family)
    # Use same regressors as the OLS model (without categorical handling via formula so that dummies are created)
    # We'll use patsy via formula but fit as GLM NB
    formula_nb = (
        'alldeaths ~ FeminineScore_z + MaxWind + MinPressure + Category + Year_c + ElapsedYears + FeminineMTurk_z + C(Source)'
    )
    # Construct design matrices
    import patsy
    y_nb, X_nb = patsy.dmatrices(formula_nb, data=df, return_type='dataframe')
    # Flatten y
    y_nb = np.asarray(y_nb).ravel()
    # Add a tiny constant to zeros is not necessary for NB, but ensure non-negative ints
    # Fit Negative Binomial via GLM
    nb_model = sm.GLM(y_nb, X_nb, family=sm.families.NegativeBinomial())
    try:
        nb_res = nb_model.fit()
    except Exception:
        # fallback: use discrete NegativeBinomial (may fail for small samples); try with start_params from OLS
        nb_res = nb_model.fit(disp=0)
    results['nb_deaths'] = nb_res

    # Model 3: OLS on log(1 + damage) as a robustness check
    formula_ols_damage = (
        'LogDamage ~ FeminineScore_z + MaxWind + MinPressure + Category + Year_c + ElapsedYears + FeminineMTurk_z + C(Source)'
    )
    ols_damage = smf.ols(formula_ols_damage, data=df).fit(cov_type='HC3')
    results['ols_logdamage'] = ols_damage

    # Return the fitted result objects for inspection (summary tables can be printed by the caller)
    return results


