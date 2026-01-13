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
    Transform the raw hurricane dataset into a modeling-ready dataframe.

    Produces the following new columns (used in modeling):
      - deaths: numeric copy of alldeaths
      - log_deaths: np.log1p(deaths)
      - log_ndam15: np.log1p(ndam15)
      - IsFemaleName: integer copy of gender_mf (0/1)
      - min_pressure: copy of 'min' renamed for clarity
      - z-scored covariates: masfem_z, wind_z, category_z, min_pressure_z, year_z, elapsedyrs_z

    Drops rows that are missing the core variables required for modeling.
    """
    df = df.copy()

    # Ensure numeric columns are numeric
    for col in ['alldeaths', 'ndam15', 'masfem', 'masfem_mturk', 'wind', 'category', 'min', 'year', 'elapsedyrs', 'gender_mf']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Rename pressure column for clarity
    if 'min' in df.columns:
        df['min_pressure'] = df['min']

    # Create primary dependent variables
    df['deaths'] = df['alldeaths']
    df['log_deaths'] = np.log1p(df['deaths'])

    # Secondary outcome: logged damage (2015-normalized)
    if 'ndam15' in df.columns:
        df['log_ndam15'] = np.log1p(df['ndam15'])
    else:
        df['log_ndam15'] = np.nan

    # Independent variable and binary female indicator
    df['masfem'] = df['masfem']
    if 'gender_mf' in df.columns:
        df['IsFemaleName'] = df['gender_mf'].astype('Int64')
    else:
        df['IsFemaleName'] = pd.NA

    # Drop rows missing essential columns for the main analysis
    required = ['deaths', 'masfem', 'wind', 'category', 'min_pressure', 'year', 'source']
    present_required = [c for c in required if c in df.columns]
    df = df.dropna(subset=present_required)

    # Compute z-scores for continuous predictors/controls (center & scale)
    def zscore(s: pd.Series) -> pd.Series:
        return (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) != 0 else 1.0)

    df['masfem_z'] = zscore(df['masfem'])
    df['wind_z'] = zscore(df['wind'])
    df['category_z'] = zscore(df['category'])
    df['min_pressure_z'] = zscore(df['min_pressure'])
    df['year_z'] = zscore(df['year'])
    if 'elapsedyrs' in df.columns:
        df['elapsedyrs_z'] = zscore(df['elapsedyrs'])
    else:
        df['elapsedyrs_z'] = pd.NA

    # Ensure 'source' is categorical (keeps as-is for patsy formula C(source))
    if 'source' in df.columns:
        df['source'] = df['source'].astype('category')

    # Keep only columns necessary for modeling to simplify downstream code
    keep_cols = [
        'deaths', 'log_deaths', 'log_ndam15',
        'masfem', 'masfem_z', 'IsFemaleName',
        'wind', 'wind_z', 'category', 'category_z',
        'min_pressure', 'min_pressure_z', 'year', 'year_z',
        'elapsedyrs_z', 'source', 'name'
    ]
    # Only keep columns that exist in df
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit two complementary models testing the association between name femininity and hurricane harm:
      1) Negative Binomial regression for death counts (deaths ~ masfem_z + controls)
      2) OLS regression for logged damage (log_ndam15 ~ masfem_z + controls)

    Returns a dict with fitted results objects: {'nb_model': nb_res, 'ols_damage': ols_res}
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    results = {}

    # Ensure that the necessary columns exist
    needed_for_nb = ['deaths', 'masfem_z', 'wind_z', 'category_z', 'min_pressure_z', 'year_z', 'IsFemaleName', 'source']
    if not all(col in df.columns for col in needed_for_nb):
        raise ValueError('Dataframe is missing one or more required columns for the main models: ' + ','.join(needed_for_nb))

    # Formula: include masfem_z (IV) and key controls; source as categorical fixed effect
    formula = 'deaths ~ masfem_z + wind_z + category_z + min_pressure_z + year_z + elapsedyrs_z + IsFemaleName + C(source)'

    # Negative binomial for count outcome (deaths)
    try:
        nb_model = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial()).fit()
        results['nb_model'] = nb_model
    except Exception as e:
        # Return the exception message for diagnostics
        results['nb_model_error'] = str(e)

    # Secondary: OLS on logged damage (continuous outcome); only fit if log_ndam15 present
    if 'log_ndam15' in df.columns and df['log_ndam15'].notna().sum() > 10:
        formula_damage = 'log_ndam15 ~ masfem_z + wind_z + category_z + min_pressure_z + year_z + elapsedyrs_z + IsFemaleName + C(source)'
        try:
            ols_damage = smf.ols(formula=formula_damage, data=df).fit()
            results['ols_damage'] = ols_damage
        except Exception as e:
            results['ols_damage_error'] = str(e)
    else:
        results['ols_damage'] = None

    # Return fitted result objects (or error messages). The caller can inspect .summary() on results that are model objects.
    return results


