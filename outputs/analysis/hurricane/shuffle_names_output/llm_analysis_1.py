from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/shuffle_names_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataset into a modeling-ready dataframe.

    Creates:
    - deaths: df['ndam15'] (count of deaths)
    - log_deaths: log1p(deaths) for OLS robustness
    - mf_score: continuous masculinity-femininity rating from column 'name' (higher = more feminine)
    - female_name: binary indicator from 'elapsedyrs' (0=male name, 1=female name)
    - max_wind: from 'wind'
    - min_pressure: from 'min'
    - saffir_category: from 'masfem' (Saffir-Simpson category 1-5 when available)
    - damage_index: from 'ind' (normalized damage measure in dataset)
    - year_observed: from 'alldeaths' (column described as the year in schema)

    Additionally standardizes continuous predictors (z-score) to aid interpretation in regression.
    Rows with missing values in the required columns are dropped.
    """

    df = df.copy()

    # Map expected columns to variables used below. Use safe numeric conversion.
    # Dependent variable
    df['deaths'] = pd.to_numeric(df.get('ndam15', pd.Series(np.nan, index=df.index)), errors='coerce')
    df['log_deaths'] = np.log1p(df['deaths'])

    # Independent variables
    # 'name' column described as the masculinity-femininity index (continuous)
    df['mf_score'] = pd.to_numeric(df.get('name', pd.Series(np.nan, index=df.index)), errors='coerce')

    # 'elapsedyrs' described as binary gender indicator (0 male, 1 female) in schema
    df['female_name'] = pd.to_numeric(df.get('elapsedyrs', pd.Series(np.nan, index=df.index)), errors='coerce')

    # Controls
    df['max_wind'] = pd.to_numeric(df.get('wind', pd.Series(np.nan, index=df.index)), errors='coerce')
    df['min_pressure'] = pd.to_numeric(df.get('min', pd.Series(np.nan, index=df.index)), errors='coerce')
    df['saffir_category'] = pd.to_numeric(df.get('masfem', pd.Series(np.nan, index=df.index)), errors='coerce')
    # normalized property damage / index
    df['damage_index'] = pd.to_numeric(df.get('ind', pd.Series(np.nan, index=df.index)), errors='coerce')
    # year observed (schema names this 'alldeaths')
    df['year_observed'] = pd.to_numeric(df.get('alldeaths', pd.Series(np.nan, index=df.index)), errors='coerce')

    # Drop rows missing the key outcome or key IV(s)
    req_cols = ['deaths', 'mf_score', 'female_name', 'max_wind', 'min_pressure', 'saffir_category', 'damage_index', 'year_observed']
    # Keep rows that have at least the dependent and one IV and core controls non-missing.
    df = df.dropna(subset=['deaths', 'mf_score', 'max_wind', 'min_pressure'])

    # Standardize continuous predictors (z-scoring). Create _z columns used in modeling.
    z_cols = {
        'mf_score': 'mf_score_z',
        'max_wind': 'max_wind_z',
        'min_pressure': 'min_pressure_z',
        'saffir_category': 'saffir_category_z',
        'damage_index': 'damage_index_z',
        'year_observed': 'year_observed_z'
    }

    for raw_col, z_col in z_cols.items():
        if raw_col in df.columns:
            vals = pd.to_numeric(df[raw_col], errors='coerce')
            # If column is entirely NaN, leave as is
            if vals.notna().sum() > 0:
                mean = vals.mean()
                std = vals.std(ddof=0) if vals.std(ddof=0) != 0 else 1.0
                df[z_col] = (vals - mean) / std
            else:
                df[z_col] = np.nan
        else:
            df[z_col] = np.nan

    # Ensure female_name is binary 0/1 (coerce nonzero to 1)
    df['female_name'] = df['female_name'].apply(lambda x: 1 if x == 1 else (0 if x == 0 else np.nan))

    # Final drop: require that the standardized IV(s) and key controls are present
    final_required = ['mf_score_z', 'female_name', 'max_wind_z', 'min_pressure_z']
    df = df.dropna(subset=['deaths', 'mf_score_z', 'max_wind_z', 'min_pressure_z'])

    # Keep and return only the columns we will use in modeling to keep the dataframe small/explicit
    keep_cols = [
        'deaths', 'log_deaths',
        'mf_score', 'mf_score_z', 'female_name',
        'max_wind', 'max_wind_z',
        'min_pressure', 'min_pressure_z',
        'saffir_category', 'saffir_category_z',
        'damage_index', 'damage_index_z',
        'year_observed', 'year_observed_z'
    ]

    # Some of these may not exist for all datasets; filter to existing columns
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit the primary Negative Binomial GLM on death counts with femininity rating and controls.
    Also fit an OLS on log_deaths as a robustness check.

    Returns a dict with fitted model results objects.
    """
    results = {}

    # Prepare data (assume df is already transformed by transform())
    modeling_df = df.copy()

    # Construct design matrix for GLM NB using continuous mf_score_z. Also run alternative model using binary female_name.
    # Common controls: max_wind_z, min_pressure_z, saffir_category_z, damage_index_z, year_observed_z
    control_vars = [c for c in [
        'max_wind_z', 'min_pressure_z', 'saffir_category_z', 'damage_index_z', 'year_observed_z'
    ] if c in modeling_df.columns]

    # Primary model 1: continuous femininity score
    from statsmodels.tools import add_constant
    X1_cols = ['mf_score_z'] + control_vars
    X1 = add_constant(modeling_df[X1_cols].astype(float), has_constant='add')
    y = modeling_df['deaths'].astype(float)

    # Fit Negative Binomial GLM (counts). Use GLM with NegativeBinomial family (can handle overdispersion).
    try:
        nb_family = sm.families.NegativeBinomial()
        nb_model = sm.GLM(y, X1, family=nb_family)
        nb_res = nb_model.fit()
        # Also obtain robust (HC3) covariances
        nb_res_robust = nb_res.get_robustcov_results(cov_type='HC3')
        results['nb_continuous_mf'] = {
            'model': nb_res,
            'robust': nb_res_robust
        }
    except Exception as e:
        results['nb_continuous_mf_error'] = str(e)

    # Primary model 2: binary female_name as predictor
    if 'female_name' in modeling_df.columns:
        X2_cols = ['female_name'] + control_vars
        X2 = add_constant(modeling_df[X2_cols].astype(float), has_constant='add')
        try:
            nb_model2 = sm.GLM(y, X2, family=nb_family)
            nb_res2 = nb_model2.fit()
            nb_res2_robust = nb_res2.get_robustcov_results(cov_type='HC3')
            results['nb_binary_female'] = {
                'model': nb_res2,
                'robust': nb_res2_robust
            }
        except Exception as e:
            results['nb_binary_female_error'] = str(e)

    # Robustness check: OLS on log_deaths
    if 'log_deaths' in modeling_df.columns:
        X_ols = add_constant(modeling_df[X1_cols].astype(float), has_constant='add')
        y_log = modeling_df['log_deaths'].astype(float)
        try:
            ols_model = sm.OLS(y_log, X_ols)
            ols_res = ols_model.fit(cov_type='HC3')
            results['ols_log_continuous_mf'] = ols_res
        except Exception as e:
            results['ols_log_continuous_mf_error'] = str(e)

    # Return the raw results objects so user can inspect .summary(), coef tables, etc.
    return results


