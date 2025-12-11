from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataset into a modeling-ready dataframe.

    Outputs (important columns):
      - Fatalities: integer count of fatalities (from 'ndam15')
      - FemininityScore: numeric name femininity index (from 'name' or fallback to FemaleNameBinary)
      - FemaleNameBinary: binary indicator of female name (from 'elapsedyrs')
      - MaxWind: maximum wind speed at landfall (from 'wind')
      - MinPressure: minimum central pressure at landfall (from 'min')
      - Category: Saffir-Simpson scale (from 'masfem')
      - PropertyDamage: normalized damage (from 'ind')
      - Year: year of the storm (from 'alldeaths')
      - logPropertyDamage: log(PropertyDamage + 1)
      - standardized versions of continuous predictors with suffix '_z'
    """
    df = df.copy()

    # Primary binary indicator: compute and fill missing -> assume not female (0)
    df['FemaleNameBinary'] = pd.to_numeric(df.get('elapsedyrs'), errors='coerce').fillna(0).astype(int)

    # Create/rename modelling columns from raw columns (handle non-numeric safely)
    df['Fatalities'] = pd.to_numeric(df.get('ndam15'), errors='coerce')
    # Attempt to parse a numeric femininity score from 'name' if provided; otherwise fallback to FemaleNameBinary
    df['FemininityScore'] = pd.to_numeric(df.get('name'), errors='coerce')
    # If FemininityScore could not be parsed numerically, use FemaleNameBinary as a conservative proxy
    df['FemininityScore'] = df['FemininityScore'].fillna(df['FemaleNameBinary'].astype(float))

    df['MaxWind'] = pd.to_numeric(df.get('wind'), errors='coerce')
    df['MinPressure'] = pd.to_numeric(df.get('min'), errors='coerce')
    df['Category'] = pd.to_numeric(df.get('masfem'), errors='coerce')
    df['PropertyDamage'] = pd.to_numeric(df.get('ind'), errors='coerce')
    # In the provided schema 'alldeaths' holds the year value
    df['Year'] = pd.to_numeric(df.get('alldeaths'), errors='coerce')

    # Drop rows missing the outcome (Fatalities) only; allow FemininityScore to be filled by fallback above
    df = df.dropna(subset=['Fatalities'])

    # Ensure Fatalities is non-negative
    df = df[df['Fatalities'] >= 0]

    # Convert fatalities to integer dtype where possible (use standard int)
    df['Fatalities'] = df['Fatalities'].astype(int)

    # Ensure FemaleNameBinary is integer (already filled above)
    df['FemaleNameBinary'] = df['FemaleNameBinary'].astype(int)

    # Create logged damage as a proxy for exposure/population affected
    df['PropertyDamage'] = df['PropertyDamage'].fillna(0)
    df['logPropertyDamage'] = np.log(df['PropertyDamage'].clip(lower=0) + 1)

    # Standardize continuous predictors for interpretability in regression
    standardize_cols = ['FemininityScore', 'MaxWind', 'MinPressure', 'Category', 'logPropertyDamage', 'Year']
    for col in standardize_cols:
        zname = col + '_z'
        if col in df.columns:
            # compute mean/std on non-missing values
            mean = df[col].mean()
            std = df[col].std(ddof=0)
            # avoid division by zero or NaN
            if pd.isna(std) or std == 0:
                df[zname] = 0.0
            else:
                df[zname] = (df[col] - mean) / std
        else:
            # If the raw column is absent entirely, create a neutral standardized column
            df[zname] = 0.0

    # Ensure required final columns exist (even if they are NaN or neutral)
    required_final = [
        'Fatalities',
        'FemininityScore_z',
        'FemaleNameBinary',
        'MaxWind_z',
        'MinPressure_z',
        'Category_z',
        'logPropertyDamage',
        'logPropertyDamage_z',
        'Year_z',
    ]
    for col in required_final:
        if col not in df.columns:
            # For standardized controls use neutral 0.0, for others use NaN
            if col.endswith('_z'):
                df[col] = 0.0
            elif col == 'logPropertyDamage':
                df[col] = 0.0
            elif col == 'FemaleNameBinary':
                df[col] = 0
            else:
                df[col] = np.nan

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit the main statistical model and a robustness OLS model.

    Primary model: Negative Binomial regression predicting Fatalities (count) from
    standardized FemininityScore and controls.

    Robustness: OLS predicting log(Fatalities + 1).

    Returns a dict with 'nb_model' and 'ols_model' fitted results objects.
    """
    df = df.copy()

    # Required columns check (must match conceptual variables)
    required = [
        'Fatalities',
        'FemininityScore_z',
        'FemaleNameBinary',
        'MaxWind_z',
        'MinPressure_z',
        'Category_z',
        'logPropertyDamage',
        'logPropertyDamage_z',
        'Year_z',
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Transformed dataframe is missing required columns: {missing}")

    # Prepare data: drop rows with missing values in model inputs (keep as many rows as possible)
    X_cols = [
        'FemininityScore_z',
        'FemaleNameBinary',
        'MaxWind_z',
        'MinPressure_z',
        'Category_z',
        'logPropertyDamage_z',
        'Year_z',
    ]
    model_df = df.dropna(subset=['Fatalities']).copy()

    # For any X_cols that are entirely missing or NaN, fill with neutral 0 to avoid dropping all rows
    for col in X_cols:
        if col not in model_df.columns:
            model_df[col] = 0.0
        else:
            # If column exists but has NaNs, fill those with 0 (neutral standardized value)
            if model_df[col].isna().any():
                model_df[col] = model_df[col].fillna(0.0)

    # After filling, ensure there is at least one row
    if model_df.shape[0] == 0:
        raise ValueError("No rows available for modeling after dropping missing values.")

    X = model_df[X_cols].astype(float)
    X = sm.add_constant(X, has_constant='add')
    y = model_df['Fatalities'].astype(float)

    # 1) Negative binomial regression for count outcome
    try:
        nb_glm = sm.GLM(y, X, family=sm.families.NegativeBinomial())
        nb_result = nb_glm.fit()
    except Exception:
        # Fall back to Poisson and compute robust covariance-adjusted results
        pois_glm = sm.GLM(y, X, family=sm.families.Poisson())
        pois_result = pois_glm.fit()
        nb_result = pois_result.get_robustcov_results(cov_type='HC3')

    # 2) Robustness: OLS on log(Fatalities + 1)
    y_log = np.log(model_df['Fatalities'] + 1)
    ols_model = sm.OLS(y_log, X)
    ols_result = ols_model.fit()
    ols_result_robust = ols_result.get_robustcov_results(cov_type='HC3')

    return {"nb_model": nb_result, "ols_model": ols_result_robust}