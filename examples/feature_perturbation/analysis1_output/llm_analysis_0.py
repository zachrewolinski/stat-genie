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
    # Make a copy
    df = df.copy()

    # Ensure expected columns exist
    expected = ['alldeaths', 'masfem', 'gender_mf', 'wind', 'min', 'category', 'year', 'elapsedyrs', 'source', 'ndam15']
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise KeyError(f"Input dataframe is missing expected columns: {missing}")

    # Rename some columns to stable names used in the model
    df = df.rename(columns={
        'min': 'MinPressure',
        'wind': 'Wind',
        'category': 'Category',
        'year': 'Year',
        'elapsedyrs': 'ElapsedYears',
        'masfem': 'masfem',
        'gender_mf': 'gender_mf',
        'alldeaths': 'alldeaths',
        'ndam15': 'ndam15',
        'source': 'source'
    })

    # Convert types
    df['Wind'] = pd.to_numeric(df['Wind'], errors='coerce')
    df['MinPressure'] = pd.to_numeric(df['MinPressure'], errors='coerce')
    df['Category'] = pd.to_numeric(df['Category'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['ElapsedYears'] = pd.to_numeric(df['ElapsedYears'], errors='coerce')
    df['masfem'] = pd.to_numeric(df['masfem'], errors='coerce')
    df['gender_mf'] = pd.to_numeric(df['gender_mf'], errors='coerce')
    df['alldeaths'] = pd.to_numeric(df['alldeaths'], errors='coerce')
    df['ndam15'] = pd.to_numeric(df['ndam15'], errors='coerce')

    # Drop rows missing any variable used in the main analysis (keeps consistent sample across models)
    df = df.dropna(subset=['alldeaths', 'masfem', 'gender_mf', 'Wind', 'MinPressure', 'Category', 'Year', 'ElapsedYears', 'source'])

    # Create dependent variable: log-transformed fatalities
    df['LogDeaths'] = np.log(df['alldeaths'] + 1)

    # Alternative/robustness dependent variable: log-transformed 2015-normalized damages
    # Note: if ndam15 was missing we already dropped above; keep consistent sample
    df['LogNDAM15'] = np.log(df['ndam15'].clip(lower=0) + 1)

    # Independent variable: mean-center masfem so coefficient reflects effect per SD-like deviation from mean
    df['NameFem'] = df['masfem'] - df['masfem'].mean()

    # Binary gender indicator (0 male, 1 female)
    df['GenderBinary'] = df['gender_mf'].astype(int)

    # Ensure source is a categorical variable (we will use categorical dummies in the model via C(Source))
    df['Source'] = df['source'].astype('category')

    # Keep only the final columns needed for modeling (makes the output explicit)
    final_cols = [
        'LogDeaths', 'LogNDAM15', 'NameFem', 'GenderBinary', 'Wind', 'MinPressure',
        'Category', 'Year', 'ElapsedYears', 'Source'
    ]
    df = df[final_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    import statsmodels.formula.api as smf

    # Main specification: effect of name femininity on log fatalities controlling for storm severity and covariates.
    # Use heteroskedasticity-robust standard errors (HC3).
    formula_main = ('LogDeaths ~ NameFem + GenderBinary + Wind + MinPressure + '
                    'Category + Year + ElapsedYears + C(Source)')
    res_main = smf.ols(formula_main, data=df).fit(cov_type='HC3')

    # Robustness 1: same specification but with log damages (ndam15) as alternative outcome
    formula_damage = ('LogNDAM15 ~ NameFem + GenderBinary + Wind + MinPressure + '
                      'Category + Year + ElapsedYears + C(Source)')
    res_damage = smf.ols(formula_damage, data=df).fit(cov_type='HC3')

    # Robustness 2: use binary gender only (replicates simplistic test that female-named storms lead to different outcomes)
    formula_genderonly = ('LogDeaths ~ GenderBinary + Wind + MinPressure + Category + Year + ElapsedYears + C(Source)')
    res_genderonly = smf.ols(formula_genderonly, data=df).fit(cov_type='HC3')

    # Return fitted result objects so the caller can inspect summaries, coefficients, and diagnostics
    return {
        'main': res_main,
        'robust_damage': res_damage,
        'gender_only': res_genderonly
    }


