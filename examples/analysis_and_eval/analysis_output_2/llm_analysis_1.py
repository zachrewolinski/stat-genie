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
    # Work on a copy
    df = df.copy()

    # Ensure required columns exist and convert to numeric where appropriate
    numeric_cols = ['alldeaths', 'masfem', 'gender_mf', 'wind', 'min', 'ndam15', 'year', 'category']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing the key variables needed for the analysis
    df = df.dropna(subset=['alldeaths', 'masfem', 'wind', 'min', 'ndam15', 'year', 'category'])

    # Dependent variable: keep count and also create logged variant for robustness
    df['alldeaths'] = df['alldeaths'].astype(int)
    df['alldeaths_log'] = np.log(df['alldeaths'] + 1)

    # Independent variables
    # Standardize continuous femininity rating
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1)
    # Binary gender indicator: rename into analytic column
    df['gender_female'] = df['gender_mf'].astype(int)

    # Controls: standardize continuous storm-strength indicators
    df['wind_z'] = (df['wind'] - df['wind'].mean()) / (df['wind'].std(ddof=0) if df['wind'].std(ddof=0) != 0 else 1)
    df['min_z'] = (df['min'] - df['min'].mean()) / (df['min'].std(ddof=0) if df['min'].std(ddof=0) != 0 else 1)

    # Logged damage (ndam15) to reduce skew
    df['ndam15_log'] = np.log(df['ndam15'] + 1)

    # Time trend
    df['year_centered'] = df['year'] - df['year'].mean()

    # Ensure category is integer (we will treat it as categorical in the model)
    df['category'] = df['category'].astype(int)

    # Keep only columns necessary for modeling + useful identifiers
    keep_cols = ['ind', 'year', 'name', 'masfem', 'masfem_z', 'gender_female', 'alldeaths', 'alldeaths_log',
                 'wind', 'wind_z', 'min', 'min_z', 'ndam15', 'ndam15_log', 'year_centered', 'category', 'source']
    # Return only existing columns from keep_cols (defensive in case some identifiers missing)
    present_cols = [c for c in keep_cols if c in df.columns]
    return df[present_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Work on a copy
    data = df.copy()

    # Primary specification: negative binomial regression on death counts
    # Formula includes the standardized femininity rating and the gender binary control,
    # and adjusts for storm strength and time trends. Category is treated as categorical.
    formula_nb = 'alldeaths ~ masfem_z + gender_female + wind_z + min_z + ndam15_log + year_centered + C(category)'

    nb_model = smf.glm(formula=formula_nb, data=data, family=sm.families.NegativeBinomial())
    nb_results = nb_model.fit()

    # Robustness: OLS on log(deaths + 1) with heteroskedasticity-robust SEs
    formula_ols = 'alldeaths_log ~ masfem_z + gender_female + wind_z + min_z + ndam15_log + year_centered + C(category)'
    ols_model = smf.ols(formula=formula_ols, data=data).fit(cov_type='HC3')

    # Return fitted results objects for downstream inspection; callers can call .summary() or examine params
    return {
        'nb_results': nb_results,
        'ols_results': ols_model
    }


