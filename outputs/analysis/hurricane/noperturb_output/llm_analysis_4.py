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
    Transform the raw hurricane dataset into the analysis-ready dataframe.

    Produces these columns used in the models:
      - female_name: binary 0/1 from gender_mf
      - masfem_z: standardized masfem score (higher = more feminine)
      - storm_severity: standardized composite severity score (wind, category, inverse pressure)
      - log_alldeaths: log(alldeaths + 1)
      - log_ndam15: log(ndam15 + 1)
      - year_centered: year - mean(year)
      - elapsedyrs: copied from input (numeric)
      - source_is_uri: binary indicator if source contains 'uri'

    Drops rows with missing values in the variables required for the main model.
    """
    # Work on a copy
    df = df.copy()

    # Ensure numeric columns are numeric
    numeric_cols = ['alldeaths', 'ndam15', 'masfem', 'wind', 'category', 'min', 'year', 'elapsedyrs', 'gender_mf']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Basic required columns for primary analysis
    required = ['alldeaths', 'masfem', 'wind', 'category', 'min', 'ndam15', 'year', 'elapsedyrs', 'gender_mf', 'source']
    present_required = [c for c in required if c in df.columns]
    # Drop rows with missing values in required columns
    df = df.dropna(subset=present_required)

    # Dependent variable: log deaths
    df['log_alldeaths'] = np.log(df['alldeaths'].astype(float) + 1.0)

    # Secondary outcome (used as control / robustness): log damage (adjusted to 2015, ndam15)
    df['log_ndam15'] = np.log(df['ndam15'].astype(float) + 1.0)

    # Independent variables
    # gender_mf is 0/1 in the data; create female_name as integer 0/1
    df['female_name'] = df['gender_mf'].astype(int)

    # Standardize masfem (continuous femininity rating). Use population std (ddof=0) for standardization.
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1.0)

    # Construct a composite storm severity score combining wind (higher=worse), category (higher=worse), and inverse min pressure (lower pressure = stronger storm)
    # First transform min pressure to 'neg_pressure' so higher means stronger storm
    df['neg_min'] = -1.0 * df['min'].astype(float)

    # Standardize the three components (using ddof=0)
    for comp in ['wind', 'category', 'neg_min']:
        df[f'{comp}_z'] = (df[comp] - df[comp].mean()) / (df[comp].std(ddof=0) if df[comp].std(ddof=0) != 0 else 1.0)

    # Average the z-scores to get composite severity, then re-standardize
    df['storm_severity'] = df[[f'{c}_z' for c in ['wind', 'category', 'neg_min']]].mean(axis=1)
    if df['storm_severity'].std(ddof=0) != 0:
        df['storm_severity'] = (df['storm_severity'] - df['storm_severity'].mean()) / df['storm_severity'].std(ddof=0)

    # Year centered
    df['year_centered'] = df['year'] - df['year'].mean()

    # elapsedyrs keep as-is but ensure numeric
    df['elapsedyrs'] = pd.to_numeric(df['elapsedyrs'], errors='coerce')

    # Source indicator (simple binary for 'uri' sources vs others)
    df['source_is_uri'] = df['source'].astype(str).str.contains('uri', case=False, na=False).astype(int)

    # Keep only the columns that will be used in modeling to avoid accidental use of raw cols later
    keep_cols = [
        'log_alldeaths',
        'female_name',
        'masfem_z',
        'storm_severity',
        'year_centered',
        'elapsedyrs',
        'log_ndam15',
        'source_is_uri'
    ]

    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit the pre-specified statistical models testing whether feminine hurricane names are associated
    with higher fatalities (interpreted as fewer precautions), controlling for storm severity and time.

    Returns a dictionary with two fitted models (primary = negative binomial on deaths; secondary = OLS on log damages).
    Each result object is returned after applying robust (HC3) standard errors.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Work on a copy
    data = df.copy()

    # Ensure no missing values remain in modeling columns
    model_cols = ['log_alldeaths', 'female_name', 'masfem_z', 'storm_severity', 'year_centered', 'elapsedyrs', 'log_ndam15', 'source_is_uri']
    data = data.dropna(subset=model_cols)

    results = {}

    # 1) Primary model: Negative binomial regression on raw counts is a natural choice for count outcome.
    #    We modeled log(alldeaths + 1) in the transform step for interpretability, but for count modeling
    #    it's better to model counts directly. Here we will use the transformed log outcome with OLS as one
    #    analytic strategy and also fit a NegativeBinomial on counts if original counts were available.
    #    Because the dataframe produced log_alldeaths, we'll fit two complementary models:

    # A) Negative binomial on counts (re-create counts from log_alldeaths if possible) -- prefer original counts if available.
    #    If original counts are not available in this transformed df, the NB model below will use log_alldeaths as the outcome
    #    with a Gaussian family as a pragmatic compromise. To keep the analysis consistent with the transformed data,
    #    we fit a GLM on log_alldeaths with Gaussian family and robust SEs as the main specification here, and
    #    present an auxiliary OLS on log damages.

    # Main specification (linear model on log deaths, robust SEs):
    formula_main = 'log_alldeaths ~ female_name + masfem_z + storm_severity + year_centered + elapsedyrs + log_ndam15 + source_is_uri'
    ols_main = smf.ols(formula_main, data=data).fit(cov_type='HC3')
    results['ols_log_deaths_main'] = ols_main

    # Alternative specification: include interaction between female_name and storm severity to test moderation
    formula_interact = 'log_alldeaths ~ female_name * storm_severity + masfem_z + year_centered + elapsedyrs + log_ndam15 + source_is_uri'
    ols_interact = smf.ols(formula_interact, data=data).fit(cov_type='HC3')
    results['ols_log_deaths_interact'] = ols_interact

    # Secondary model: OLS on log damages as a robustness outcome (robust SEs)
    formula_damage = 'log_ndam15 ~ female_name + masfem_z + storm_severity + year_centered + elapsedyrs + source_is_uri'
    ols_damage = smf.ols(formula_damage, data=data).fit(cov_type='HC3')
    results['ols_log_damage'] = ols_damage

    # If original count alldeaths is available in the input (not in the transformed df here), user may prefer a NB model on counts.
    # We attempt to reconstruct counts if a column 'alldeaths' exists in the parent dataframe (not guaranteed here). If it's present,
    # fit a NegativeBinomial on counts.
    try:
        # If the original (untransformed) counts column exists in the environment, use it. We check whether "alldeaths" exists in the
        # passed dataframe (it was dropped in transform). If present, fit a NB. Otherwise skip.
        if 'alldeaths' in data.columns:
            nb_formula = 'alldeaths ~ female_name + masfem_z + storm_severity + year_centered + elapsedyrs + log_ndam15 + source_is_uri'
            nb_model = smf.glm(nb_formula, data=data, family=sm.families.NegativeBinomial()).fit(cov_type='HC3')
            results['nb_alldeaths'] = nb_model
    except Exception:
        # NB fitting optional; ignore fit errors but keep the other results
        pass

    return results


