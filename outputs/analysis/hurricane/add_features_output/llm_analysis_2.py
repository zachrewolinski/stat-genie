from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/add_features_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Create IVs (keep original measures but put them into clean columns)
    df['FemScore'] = pd.to_numeric(df.get('masfem', None), errors='coerce')
    # gender_mf is coded 0/1 in the raw data; keep as numeric (nullable integer)
    df['FemaleBinary'] = pd.to_numeric(df.get('gender_mf', None), errors='coerce')

    # Dependent variables: use log(1 + alldeaths) and log(1 + ndam15) as proxies
    df['LogDeaths'] = np.log1p(pd.to_numeric(df.get('alldeaths', None), errors='coerce'))
    df['LogDamage2015'] = np.log1p(pd.to_numeric(df.get('ndam15', None), errors='coerce'))

    # Controls: storm intensity measures and time covariates
    df['WindSpeed'] = pd.to_numeric(df.get('wind', None), errors='coerce')
    df['MinPressure'] = pd.to_numeric(df.get('min', None), errors='coerce')
    df['Category'] = pd.to_numeric(df.get('category', None), errors='coerce')
    df['ElapsedYears'] = pd.to_numeric(df.get('elapsedyrs', None), errors='coerce')
    df['Year'] = pd.to_numeric(df.get('year', None), errors='coerce')

    # Construct a composite intensity z-score: higher = more intense (high wind, low pressure)
    # z(wind) - z(pressure) (since lower pressure = stronger storm)
    # Use population std (ddof=0) to avoid small-sample ddof effects
    if df['WindSpeed'].notna().sum() > 0:
        wind_mean = df['WindSpeed'].mean()
        wind_std = df['WindSpeed'].std(ddof=0)
    else:
        wind_mean = np.nan
        wind_std = np.nan
    if df['MinPressure'].notna().sum() > 0:
        press_mean = df['MinPressure'].mean()
        press_std = df['MinPressure'].std(ddof=0)
    else:
        press_mean = np.nan
        press_std = np.nan

    # Avoid division by zero
    df['Wind_z'] = (df['WindSpeed'] - wind_mean) / (wind_std if wind_std and not np.isclose(wind_std, 0) else np.nan)
    df['Pressure_z'] = (df['MinPressure'] - press_mean) / (press_std if press_std and not np.isclose(press_std, 0) else np.nan)
    df['Intensity_z'] = df['Wind_z'] - df['Pressure_z']

    # Make Category an integer where possible (useful for C(Category) in formulas)
    df['Category'] = df['Category'].astype('Int64')

    # Drop rows that are missing the primary DV or IV or key controls used in the main specification
    df = df.dropna(subset=['LogDeaths', 'FemScore', 'Intensity_z', 'Category', 'ElapsedYears'])

    # Final columns retained for modeling (explicitly listed for clarity)
    keep_cols = [
        'FemScore', 'FemaleBinary', 'LogDeaths', 'LogDamage2015',
        'WindSpeed', 'MinPressure', 'Wind_z', 'Pressure_z', 'Intensity_z',
        'Category', 'ElapsedYears', 'Year',
        # retain raw fields that may be useful for follow-up checks
        'name', 'masfem', 'gender_mf', 'alldeaths', 'ndam15'
    ]

    # If any of the keep_cols are missing from df (e.g., because original didn't contain them), still return df
    existing_keep = [c for c in keep_cols if c in df.columns]
    return df[existing_keep]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    import statsmodels.formula.api as smf
    # Work on a copy to avoid in-place modifications
    df = df.copy()

    # Main specification: OLS on log deaths (continuous proxy), using robust (HC3) standard errors.
    # We control for storm intensity (composite z), category (categorical), and elapsed years.
    # Formula uses FemScore (continuous femininity); we also fit a binary-name specification for robustness.

    # Ensure Category is treated as categorical in the formula
    df['Category'] = df['Category'].astype('category')

    # Primary model: FemScore -> LogDeaths
    formula_main = 'LogDeaths ~ FemScore + Intensity_z + C(Category) + ElapsedYears'
    model_main = smf.ols(formula_main, data=df).fit(cov_type='HC3')

    # Robustness 1: use binary female name indicator instead of continuous femininity
    formula_bin = 'LogDeaths ~ FemaleBinary + Intensity_z + C(Category) + ElapsedYears'
    model_bin = smf.ols(formula_bin, data=df).fit(cov_type='HC3')

    # Robustness 2: use economic damage (log(ndam15+1)) as an alternative dependent variable
    # Keep only rows with LogDamage2015 available
    df_damage = df.dropna(subset=['LogDamage2015'])
    formula_damage = 'LogDamage2015 ~ FemScore + Intensity_z + C(Category) + ElapsedYears'
    model_damage = smf.ols(formula_damage, data=df_damage).fit(cov_type='HC3')

    # Print summaries for quick inspection (can be removed if not desired)
    print('\n=== Main model (LogDeaths ~ FemScore + controls) ===')
    print(model_main.summary())
    print('\n=== Binary-name robustness (LogDeaths ~ FemaleBinary + controls) ===')
    print(model_bin.summary())
    print('\n=== Damage robustness (LogDamage2015 ~ FemScore + controls) ===')
    print(model_damage.summary())

    # Return model result objects so caller can inspect coefficients, p-values, etc.
    return {
        'main_deaths_model': model_main,
        'binary_deaths_model': model_bin,
        'damage_model': model_damage
    }


