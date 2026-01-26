from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/replace_with_rvs_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Ensure numeric conversions for key columns
    numeric_cols = ['masfem', 'masfem_mturk', 'wind', 'min', 'category', 'alldeaths', 'ndam15', 'year', 'elapsedyrs', 'gender_mf']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows with missing essential variables for the primary model
    required_for_primary = ['alldeaths', 'masfem', 'wind', 'min', 'category', 'year', 'elapsedyrs']
    df = df.dropna(subset=required_for_primary)

    # Create primary DV and a logged alternative for diagnostics
    df['alldeaths'] = df['alldeaths'].astype(int)
    df['log1p_alldeaths'] = np.log1p(df['alldeaths'])

    # Create log-transformed damage variable (alternative DV for robustness)
    if 'ndam15' in df.columns:
        df['ndam15'] = pd.to_numeric(df['ndam15'], errors='coerce')
        df['log1p_ndam15'] = np.log1p(df['ndam15'].fillna(0))

    # Binary female name indicator (ensure integer 0/1)
    if 'gender_mf' in df.columns:
        df['FemaleName'] = df['gender_mf'].fillna(0).astype(int)
    else:
        df['FemaleName'] = 0

    # Standardize continuous predictors for interpretability in models
    def standardize(series: pd.Series, new_name: str) -> None:
        m = series.mean()
        s = series.std(ddof=0)
        if s == 0 or np.isnan(s):
            df[new_name] = series - m
        else:
            df[new_name] = (series - m) / s

    standardize(df['masfem'], 'masfem_s')

    if 'masfem_mturk' in df.columns:
        df['masfem_mturk'] = pd.to_numeric(df['masfem_mturk'], errors='coerce')
        df['masfem_mturk'] = df['masfem_mturk'].fillna(df['masfem_mturk'].mean())
        standardize(df['masfem_mturk'], 'masfem_mturk_s')
    else:
        df['masfem_mturk_s'] = 0.0

    standardize(df['wind'], 'wind_s')
    standardize(df['min'], 'min_s')
    standardize(df['year'], 'year_s')
    standardize(df['elapsedyrs'], 'elapsedyrs_s')

    # Ensure category numeric (use as numeric categorical predictor)
    df['category'] = pd.to_numeric(df['category'], errors='coerce')

    # Create dummies for data source (drop first to avoid multicollinearity)
    if 'source' in df.columns:
        src_dummies = pd.get_dummies(df['source'].astype(str), prefix='source', drop_first=False)
        # Keep all dummies (we will include all and rely on statsmodels to drop collinearity if needed),
        # but to follow a clear approach we will drop the most common category to avoid perfect multicollinearity.
        # Choose to drop the first column alphabetically to be deterministic.
        if src_dummies.shape[1] > 0:
            drop_col = sorted(list(src_dummies.columns))[0]
            src_dummies = src_dummies.drop(columns=[drop_col])
        df = pd.concat([df, src_dummies], axis=1)
    else:
        # If no source column, create no-op
        pass

    # Keep only columns relevant for modeling and diagnostics to return
    # (but preserve other columns in case user wants them)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Work with a copy of transformed df
    df = df.copy()

    # Identify source dummies that were created in transform
    source_dummies = [c for c in df.columns if c.startswith('source_')]

    # Base covariates
    covariates = [
        'masfem_s',          # primary IV (standardized perceived femininity)
        'masfem_mturk_s',    # robustness control (MTurk perceived femininity standardized)
        'FemaleName',        # alternative IV (binary female name indicator)
        'wind_s',
        'min_s',
        'category',
        'year_s',
        'elapsedyrs_s'
    ]

    # Add any source dummies
    covariates += source_dummies

    # Build formula for the negative binomial model on death counts
    formula_nb = 'alldeaths ~ ' + ' + '.join(covariates)

    # Fit Negative Binomial (GLM) for count DV (alldeaths)
    # Use robust covariance (HC3) to protect against heteroskedasticity
    nb_model = smf.glm(formula=formula_nb, data=df, family=sm.families.NegativeBinomial()).fit(cov_type='HC3')

    # Secondary robustness: OLS on log(1 + deaths)
    formula_ols = 'log1p_alldeaths ~ ' + ' + '.join(covariates)
    ols_model = smf.ols(formula=formula_ols, data=df).fit(cov_type='HC3')

    # Additional robustness: OLS on logged damage (if available)
    damage_result = None
    if 'log1p_ndam15' in df.columns:
        formula_dam = 'log1p_ndam15 ~ ' + ' + '.join(covariates)
        damage_model = smf.ols(formula=formula_dam, data=df).fit(cov_type='HC3')
        damage_result = damage_model

    # Return fitted results objects for downstream inspection
    return {
        'nb_model': nb_model,
        'ols_log_deaths': ols_model,
        'ols_log_damage': damage_result,
        'formula_nb': formula_nb,
        'formula_ols': formula_ols,
        'used_covariates': covariates
    }


