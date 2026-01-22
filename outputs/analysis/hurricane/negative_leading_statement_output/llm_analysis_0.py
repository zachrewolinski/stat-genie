from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/negative_leading_statement_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataframe into analysis-ready dataframe.

    Produces:
    - LogDeaths: log1p(alldeaths)
    - LogDamage: log1p(ndam15) (kept for robustness checks)
    - masfem_std: standardized masfem (z-score)
    - masfem_mturk_std: standardized masfem_mturk (z-score)
    - year_c: centered year (year - mean(year))
    - source_* dummies: cleaned source indicator dummies (source_uri, source_wiki, source_mwr, source_other)

    Keeps essential controls: wind, min, category, gender_mf, elapsedyrs
    """

    df = df.copy()

    # Ensure numeric columns are numeric
    numeric_cols = ['masfem', 'masfem_mturk', 'min', 'wind', 'category', 'alldeaths', 'ndam15', 'gender_mf', 'elapsedyrs', 'year']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Basic drop: need masfem (IV) and alldeaths (DV) and core controls to run primary model
    df = df.dropna(subset=['masfem', 'alldeaths', 'wind', 'min', 'category', 'year'])

    # Outcome transformations
    df['LogDeaths'] = np.log1p(df['alldeaths'])
    # keep damages (alternative outcome) also log-transformed for robustness
    if 'ndam15' in df.columns:
        df['LogDamage'] = np.log1p(df['ndam15'])
    else:
        df['LogDamage'] = np.nan

    # Standardize femininity indexes (z-scores)
    df['masfem_std'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1)
    if 'masfem_mturk' in df.columns:
        df['masfem_mturk_std'] = (df['masfem_mturk'] - df['masfem_mturk'].mean()) / (df['masfem_mturk'].std(ddof=0) if df['masfem_mturk'].std(ddof=0) != 0 else 1)
    else:
        df['masfem_mturk_std'] = np.nan

    # Center year to aid interpretation and minimize collinearity
    df['year_c'] = df['year'] - df['year'].mean()

    # Clean source strings and create dummies with safe column names
    if 'source' in df.columns:
        src = df['source'].astype(str).str.lower()
        src_clean = pd.Series('other', index=src.index)
        src_clean.loc[src.str.contains('uri', na=False)] = 'uri'
        src_clean.loc[src.str.contains('wiki', na=False)] = 'wiki'
        src_clean.loc[src.str.contains('mwr', na=False)] = 'mwr'
        df['source_clean'] = src_clean
        src_dummies = pd.get_dummies(df['source_clean'], prefix='source')
        # Ensure all expected columns exist (uri, wiki, mwr, other)
        for col in ['source_uri', 'source_wiki', 'source_mwr', 'source_other']:
            if col not in src_dummies.columns:
                src_dummies[col] = 0
        # concat dummies
        df = pd.concat([df, src_dummies[['source_uri', 'source_wiki', 'source_mwr', 'source_other']]], axis=1)
    else:
        # if no source, create zeros
        df['source_uri'] = 0
        df['source_wiki'] = 0
        df['source_mwr'] = 0
        df['source_other'] = 0

    # Keep and re-order columns relevant for modeling
    keep_cols = [
        'LogDeaths', 'LogDamage',
        'masfem_std', 'masfem_mturk_std', 'gender_mf',
        'wind', 'min', 'category', 'year_c', 'elapsedyrs',
        'source_uri', 'source_wiki', 'source_mwr', 'source_other'
    ]
    # Some columns might not exist (e.g., masfem_mturk_std); ensure present
    for col in keep_cols:
        if col not in df.columns:
            df[col] = np.nan

    df_final = df[keep_cols].reset_index(drop=True)
    return df_final


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Run the main specification and robustness checks.

    Specifications:
    1) Main: OLS of LogDeaths on masfem_std controlling for wind, min, category, year_c, elapsedyrs, gender_mf and source dummies. Robust (HC3) standard errors.
    2) Robustness A: replace masfem_std with masfem_mturk_std (alternative femininity measure) if available.
    3) Robustness B: use LogDamage (log1p(ndam15)) as the dependent variable with the same covariates.

    Returns a dict of fitted statsmodels results objects (or summaries if fit fails).
    """

    import statsmodels.formula.api as smf
    results = {}

    # Build base formula components
    covariates = ['wind', 'min', 'category', 'year_c', 'elapsedyrs', 'gender_mf', 'source_uri', 'source_wiki', 'source_mwr', 'source_other']
    cov_str = ' + '.join(covariates)

    # 1) Main model: LogDeaths ~ masfem_std + controls
    formula_main = 'LogDeaths ~ masfem_std + ' + cov_str
    try:
        mod1 = smf.ols(formula_main, data=df).fit(cov_type='HC3')
        results['main_masfem_on_deaths'] = mod1
    except Exception as e:
        results['main_masfem_on_deaths_error'] = str(e)

    # 2) Robustness A: masfem_mturk_std if available
    if 'masfem_mturk_std' in df.columns and df['masfem_mturk_std'].notna().sum() > 10:
        formula_mturk = 'LogDeaths ~ masfem_mturk_std + ' + cov_str
        try:
            mod2 = smf.ols(formula_mturk, data=df).fit(cov_type='HC3')
            results['robust_mturk_on_deaths'] = mod2
        except Exception as e:
            results['robust_mturk_on_deaths_error'] = str(e)
    else:
        results['robust_mturk_on_deaths'] = 'masfem_mturk_std not available or insufficient non-missing observations'

    # 3) Robustness B: outcome = LogDamage
    if 'LogDamage' in df.columns and df['LogDamage'].notna().sum() > 10:
        formula_damage = 'LogDamage ~ masfem_std + ' + cov_str
        try:
            mod3 = smf.ols(formula_damage, data=df).fit(cov_type='HC3')
            results['masfem_on_damage'] = mod3
        except Exception as e:
            results['masfem_on_damage_error'] = str(e)
    else:
        results['masfem_on_damage'] = 'LogDamage not available or insufficient non-missing observations'

    # Return the fitted model objects so the caller can inspect coefficients, p-values and summaries.
    return results


