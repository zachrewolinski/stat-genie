from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/anonymize_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw Simonsohn hurricane dataset into analysis-ready dataframe.

    Produces standardized predictors and logged outcomes used by the model function.
    Important final columns returned (names must match those referenced in the model):
      - storm_id, year, name, masfem, female_name, max_wind, min_pressure, category, deaths,
        damage_2015, log_deaths, log_damage_z, masfem_z, max_wind_z, min_pressure_z,
        year_centered, source
    """
    df = df.copy()

    # Rename the generic feature columns to meaningful variable names
    rename_map = {
        'feature1': 'storm_id',
        'feature2': 'year',
        'feature3': 'name',
        'feature4': 'masfem',            # continuous masfem index from independent coders
        'feature5': 'min_pressure',      # minimum pressure at landfall
        'feature6': 'female_name',       # binary indicator: 0 male, 1 female name
        'feature7': 'category',          # Saffir-Simpson category
        'feature8': 'deaths',            # total deaths
        'feature9': 'damage_unadj',      # unadjusted damage (various columns exist)
        'feature10': 'years_since',      # years since storm
        'feature11': 'source',           # data source
        'feature12': 'mturk_masfem',     # MTurk masfem rating
        'feature13': 'max_wind',         # maximum wind speed at landfall
        'feature14': 'damage_2015'       # damage adjusted to 2015 values (preferred damage measure)
    }
    df.rename(columns=rename_map, inplace=True)

    # Keep only rows with key variables present (deaths and masfem and max_wind are essential)
    df = df.dropna(subset=['deaths', 'masfem', 'max_wind'])

    # Ensure numeric columns are numeric
    numeric_cols = ['masfem', 'min_pressure', 'female_name', 'category', 'deaths', 'damage_2015', 'max_wind', 'year']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # After coercion, drop any rows that became NA for the essential numeric fields
    df = df.dropna(subset=['deaths', 'masfem', 'max_wind', 'year'])

    # Outcome transforms
    df['log_deaths'] = np.log1p(df['deaths'].astype(float))

    # Damage: prefer damage_2015 if present, otherwise fall back to damage_unadj
    if 'damage_2015' in df.columns and df['damage_2015'].notna().any():
        df['damage_for_model'] = df['damage_2015']
    else:
        df['damage_for_model'] = df.get('damage_unadj', np.nan)

    df['log_damage'] = np.log1p(pd.to_numeric(df['damage_for_model'].fillna(0).astype(float)))

    # Standardize continuous predictors (z-scores). Use population std (ddof=0) for stability on small samples.
    def z(x):
        x = pd.to_numeric(x, errors='coerce')
        return (x - x.mean()) / x.std(ddof=0)

    df['masfem_z'] = z(df['masfem'])
    df['max_wind_z'] = z(df['max_wind'])
    df['min_pressure_z'] = z(df['min_pressure'])
    df['log_damage_z'] = z(df['log_damage'])

    # Ensure female_name is integer 0/1
    df['female_name'] = df['female_name'].astype(int)

    # Center year to aid interpretability
    df['year_centered'] = df['year'] - df['year'].mean()

    # Keep only columns needed for modeling (and a few originals for reference)
    keep_cols = [
        'storm_id', 'year', 'year_centered', 'name', 'masfem', 'masfem_z', 'female_name',
        'max_wind', 'max_wind_z', 'min_pressure', 'min_pressure_z', 'category', 'deaths', 'log_deaths',
        'damage_for_model', 'log_damage', 'log_damage_z', 'source'
    ]
    # Some columns may not exist in every variant; filter to existing
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit primary and robustness models to test whether more-feminine hurricane names are associated
    with fewer fatalities (a proxy for lower precaution/underestimation of risk).

    Primary specification: Negative Binomial regression of raw death counts on standardized masfem
    controlling for storm intensity and damage and year trends. Negative binomial is used because
    deaths are count data with overdispersion.

    Robustness: OLS on log(1 + deaths) with heteroskedasticity-robust standard errors, and a model
    that uses the binary female_name indicator.

    Returns a dict with fitted model objects and summary text.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    results = {}
    dfm = df.copy()

    # Create categorical dummies for Saffir-Simpson category (drop first to avoid multicollinearity)
    if 'category' in dfm.columns:
        cat_dummies = pd.get_dummies(dfm['category'].astype('category'), prefix='cat', drop_first=True)
        dfm = pd.concat([dfm, cat_dummies], axis=1)
    else:
        cat_dummies = pd.DataFrame(index=dfm.index)

    # Build predictor list
    base_predictors = ['masfem_z', 'max_wind_z', 'min_pressure_z', 'log_damage_z', 'year_centered']
    # Add category dummies if present
    predictors = [p for p in base_predictors if p in dfm.columns] + list(cat_dummies.columns)

    # Ensure there are no NA values in model variables
    model_df = dfm.dropna(subset=predictors + ['deaths', 'log_deaths'])

    X = model_df[predictors]
    X = sm.add_constant(X)
    y_counts = model_df['deaths'].astype(float)
    y_log = model_df['log_deaths'].astype(float)

    # 1) Negative binomial (counts model) - primary
    try:
        nb = sm.GLM(y_counts, X, family=sm.families.NegativeBinomial()).fit()
        results['nb_model'] = nb
        results['nb_summary'] = nb.summary().as_text()
    except Exception as e:
        results['nb_error'] = str(e)

    # 2) OLS on log(1 + deaths) with robust SEs - robustness
    try:
        ols = sm.OLS(y_log, X).fit(cov_type='HC3')
        results['ols_model'] = ols
        results['ols_summary'] = ols.summary().as_text()
    except Exception as e:
        results['ols_error'] = str(e)

    # 3) Alternate specification using the binary female_name indicator (NB)
    if 'female_name' in model_df.columns:
        alt_predictors = [p for p in ['female_name', 'max_wind_z', 'min_pressure_z', 'log_damage_z', 'year_centered'] if p in model_df.columns] + list(cat_dummies.columns)
        X2 = sm.add_constant(model_df[alt_predictors])
        try:
            nb_alt = sm.GLM(model_df['deaths'].astype(float), X2, family=sm.families.NegativeBinomial()).fit()
            results['nb_alt_model'] = nb_alt
            results['nb_alt_summary'] = nb_alt.summary().as_text()
        except Exception as e:
            results['nb_alt_error'] = str(e)

        try:
            ols_alt = sm.OLS(model_df['log_deaths'].astype(float), X2).fit(cov_type='HC3')
            results['ols_alt_model'] = ols_alt
            results['ols_alt_summary'] = ols_alt.summary().as_text()
        except Exception as e:
            results['ols_alt_error'] = str(e)

    # Add a short diagnostic: mean and variance of deaths to check overdispersion
    results['deaths_mean'] = float(y_counts.mean())
    results['deaths_var'] = float(y_counts.var())

    return results


