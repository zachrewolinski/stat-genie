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
    Transform the raw Simonsohn et al. hurricane dataset to a modeling-ready dataframe.

    Creates/renames columns used in the statistical model (see conceptual variables):
      - MasFemIndex -> MasFemIndex (continuous), then standardized to MasFem_z
      - FemaleName -> FemaleName (0/1)
      - Deaths -> Deaths (count DV)
      - MaxWindMPH, MinPressure, Category, Year -> used as controls
      - Damage2015 -> LogDamage2015 (log1p)

    Drops rows missing essential modeling columns.

    Returns the transformed dataframe with all final columns.
    """
    df = df.copy()

    # Rename columns from the provided schema to meaningful names used downstream
    df = df.rename(columns={
        'feature1': 'ID',
        'feature2': 'Year',
        'feature3': 'Name',
        'feature4': 'MasFemIndex',
        'feature5': 'MinPressure',
        'feature6': 'FemaleName',
        'feature7': 'Category',
        'feature8': 'Deaths',
        'feature9': 'DamageRaw',
        'feature10': 'YearsSince',
        'feature11': 'Source',
        'feature12': 'MTurkMasFem',
        'feature13': 'MaxWindMPH',
        'feature14': 'Damage2015'
    })

    # Ensure numeric types where expected (coerce invalid -> NaN)
    numeric_cols = [
        'MasFemIndex', 'MinPressure', 'FemaleName', 'Category', 'Deaths', 'DamageRaw',
        'YearsSince', 'MTurkMasFem', 'MaxWindMPH', 'Damage2015', 'Year'
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Source as categorical
    if 'Source' in df.columns:
        df['Source'] = df['Source'].astype('category')

    # Drop rows missing essential variables (DV and main IV and key severity controls)
    required = ['MasFemIndex', 'Deaths', 'MaxWindMPH', 'Category', 'Year']
    missing_req = [c for c in required if c not in df.columns]
    if missing_req:
        raise ValueError('Missing required columns in input dataframe: %s' % missing_req)

    df = df.dropna(subset=required)

    # Ensure binary variable is integer 0/1
    # The original coding in schema is 0/1; coerce to int after dropping NA
    df['FemaleName'] = df['FemaleName'].astype(int)

    # Create log-transformed damage (2015 normalized) used as a severity control
    # If Damage2015 missing but DamageRaw present, prefer Damage2015; otherwise use DamageRaw
    if 'Damage2015' not in df.columns or df['Damage2015'].isna().all():
        if 'DamageRaw' in df.columns:
            df['Damage2015'] = df['DamageRaw']
        else:
            df['Damage2015'] = np.nan
    df['LogDamage2015'] = np.log1p(df['Damage2015'].fillna(0.0))

    # DV transformation for inspection (we model counts directly with NB) but keep log for diagnostics
    df['LogDeaths'] = np.log1p(df['Deaths'])

    # Standardize the MasFem index (z-score) to aid interpretation/stability
    df['MasFem_z'] = (df['MasFemIndex'] - df['MasFemIndex'].mean()) / (df['MasFemIndex'].std(ddof=0) if df['MasFemIndex'].std(ddof=0) != 0 else 1.0)

    # Mean-center year to improve interpretability
    df['Year_c'] = df['Year'] - df['Year'].mean()

    # Treat Category as categorical (Saffir-Simpson categories)
    df['Category'] = df['Category'].astype('category')

    # Final column checklist (these are used in the model code / conceptual variables)
    final_cols = [
        'ID', 'Year', 'Year_c', 'Name', 'MasFemIndex', 'MasFem_z', 'FemaleName',
        'MinPressure', 'MaxWindMPH', 'Category', 'Deaths', 'LogDeaths', 'Damage2015', 'LogDamage2015', 'Source'
    ]
    for c in final_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Return only rows used for modeling (we already dropped rows missing required columns)
    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a negative-binomial generalized linear model predicting hurricane fatalities (Deaths)
    from the femininity of the hurricane name and relevant controls.

    Model formula:
      Deaths ~ MasFem_z + FemaleName + MaxWindMPH + MinPressure + C(Category) + Year_c + LogDamage2015 + C(Source)

    Returns the fitted model results object (statsmodels result) so callers can inspect .summary(), params, conf_int(), etc.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Check required model columns exist
    required = [
        'Deaths', 'MasFem_z', 'FemaleName', 'MaxWindMPH', 'MinPressure', 'Category', 'Year_c', 'LogDamage2015', 'Source'
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError('Missing required columns for modeling: %s' % missing)

    # Drop rows with NA in any model column
    model_df = df.dropna(subset=required).copy()

    # If the outcome has extremely small variance (unlikely), fall back to OLS on log deaths
    if model_df['Deaths'].nunique() <= 1:
        # fallback OLS on LogDeaths
        if 'LogDeaths' not in model_df.columns:
            model_df['LogDeaths'] = np.log1p(model_df['Deaths'])
        formula_ols = 'LogDeaths ~ MasFem_z + FemaleName + MaxWindMPH + MinPressure + C(Category) + Year_c + LogDamage2015 + C(Source)'
        ols_res = smf.ols(formula=formula_ols, data=model_df).fit()
        return ols_res

    # Primary model: Negative Binomial GLM for count outcome with potential overdispersion
    formula_nb = 'Deaths ~ MasFem_z + FemaleName + MaxWindMPH + MinPressure + C(Category) + Year_c + LogDamage2015 + C(Source)'

    # Fit the GLM negative binomial; statsmodels will estimate the NB dispersion parameter
    nb_model = smf.glm(formula=formula_nb, data=model_df, family=sm.families.NegativeBinomial()).fit()

    # Return the fitted model result object
    return nb_model


