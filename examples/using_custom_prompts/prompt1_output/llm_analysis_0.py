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
    Transform the raw hurricane dataframe into the analytic dataframe used for modeling.

    Produces the following columns (at minimum):
      - LogDeaths: log1p(alldeaths)
      - MasFem_z: z-scored 'masfem' (primary IV)
      - MasFem_mturk_z: z-scored 'masfem_mturk' (alternative name-femininity measure, kept for robustness)
      - GenderF: binary gender indicator copied from 'gender_mf'
      - LogDamage: log1p(ndam15)
      - Wind_z: z-scored 'wind'
      - Year_c: year centered
      - plus original columns used as controls (category, min, source, ind, name)
    """
    df = df.copy()

    # Ensure numeric columns are numeric where appropriate
    cols_to_numeric = ['masfem', 'alldeaths', 'wind', 'category', 'min', 'ndam15', 'year', 'masfem_mturk', 'gender_mf', 'elapsedyrs']
    for c in cols_to_numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing primary variables needed for the main analysis
    required = [c for c in ['masfem', 'alldeaths', 'wind', 'category', 'ndam15', 'min', 'year'] if c in df.columns]
    df = df.dropna(subset=required)

    # Dependent variable: log(1 + deaths)
    df['LogDeaths'] = np.log1p(df['alldeaths'])

    # Main independent variable: standardized femininity rating
    df['MasFem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1)

    # Alternative femininity measure from MTurk (if present)
    if 'masfem_mturk' in df.columns:
        df['MasFem_mturk_z'] = (df['masfem_mturk'] - df['masfem_mturk'].mean()) / (df['masfem_mturk'].std(ddof=0) if df['masfem_mturk'].std(ddof=0) != 0 else 1)
    else:
        df['MasFem_mturk_z'] = np.nan

    # Binary gender indicator (0 male, 1 female)
    if 'gender_mf' in df.columns:
        df['GenderF'] = df['gender_mf'].astype(int)
    else:
        df['GenderF'] = np.nan

    # Damage control: log-transformed normalized damage (2015 dollars available in ndam15)
    if 'ndam15' in df.columns:
        df['LogDamage'] = np.log1p(df['ndam15'])
    else:
        df['LogDamage'] = np.nan

    # Standardize wind
    df['Wind_z'] = (df['wind'] - df['wind'].mean()) / (df['wind'].std(ddof=0) if df['wind'].std(ddof=0) != 0 else 1)

    # Year centered
    df['Year_c'] = df['year'] - df['year'].mean()

    # Keep a conservative set of columns used in modeling and reporting
    keep_cols = []
    for c in ['ind', 'year', 'name', 'masfem', 'MasFem_z', 'masfem_mturk', 'MasFem_mturk_z', 'gender_mf', 'GenderF', 'alldeaths', 'LogDeaths', 'wind', 'Wind_z', 'category', 'min', 'ndam15', 'LogDamage', 'elapsedyrs', 'source', 'Year_c']:
        if c in df.columns:
            keep_cols.append(c)

    return df[keep_cols].copy()


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Run the primary statistical models testing whether more-feminine hurricane names predict fewer fatalities
    (used as a proxy for fewer precautionary measures), controlling for objective storm severity and exposure.

    Returns a dict with:
      - 'ols_robust': OLS on LogDeaths with HC3 robust SEs
      - 'nb_glm': Negative-binomial GLM on raw death counts (sensitivity check)
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    results = {}

    # Ensure required columns are present
    required = ['LogDeaths', 'MasFem_z', 'Wind_z', 'category', 'LogDamage', 'min', 'Year_c', 'source', 'alldeaths']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Primary model: OLS on log-deaths with robust (HC3) standard errors.
    # Include source as a categorical control to absorb differences in provenance.
    formula = 'LogDeaths ~ MasFem_z + Wind_z + category + LogDamage + min + Year_c + C(source)'
    ols_model = smf.ols(formula, data=df).fit(cov_type='HC3')
    results['ols_robust'] = ols_model

    # Sensitivity: Negative Binomial GLM on raw counts (allows count model and overdispersion).
    # Build the design matrix X corresponding to key predictors (exclude source dummies for NB to keep X concise).
    X_cols = ['MasFem_z', 'Wind_z', 'category', 'LogDamage', 'min', 'Year_c']
    X = df[X_cols].copy()
    X = sm.add_constant(X)
    # Fit GLM negative binomial with log link (default for NB in statsmodels)
    nb_glm = sm.GLM(df['alldeaths'], X, family=sm.families.NegativeBinomial()).fit()
    results['nb_glm'] = nb_glm

    # Optionally: include a robustness OLS replacing MasFem_z with binary gender indicator
    if 'GenderF' in df.columns and df['GenderF'].notna().all():
        formula_bin = 'LogDeaths ~ GenderF + Wind_z + category + LogDamage + min + Year_c + C(source)'
        results['ols_gender_binary'] = smf.ols(formula_bin, data=df).fit(cov_type='HC3')

    return results


