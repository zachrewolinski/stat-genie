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
    Transform the raw archival hurricane dataframe into the analysis-ready dataframe.

    - Rename relevant columns to descriptive names used in modeling.
    - Convert types, create log-transformed property damage, z-score continuous covariates and the name femininity index,
      center Year, and set categorical columns.
    - Drop rows with missing values in key columns required for the primary analysis (Fatalities and name/gender measures and core controls).

    Returns the transformed dataframe containing the columns referenced in the conceptual variables.
    """
    df = df.copy()

    # Rename raw feature columns to descriptive names used in modeling
    rename_map = {
        'feature12': 'NameIsFemale',        # binary 0 male, 1 female
        'feature9': 'NameFemininity',       # coder masculinity-femininity index (continuous)
        'feature13': 'Fatalities',          # total number of deaths
        'feature8': 'PropertyDamage',       # property damage normalized to 2015 values (continuous)
        'feature7': 'WindSpeed',            # maximum wind speed at landfall
        'feature4': 'Category',             # Saffir-Simpson category
        'feature5': 'Year',                 # year of storm
        'feature14': 'MinPressure',         # minimum central pressure at landfall
        'feature10': 'Source',              # data source (categorical)
        'feature2': 'StormID',              # unique id for each storm
        'feature11': 'NameFemininity_MTURK',# MTurk masculinity-femininity index (robustness)
        'feature3': 'YearsSince'            # years elapsed since the hurricane (not used primary)
    }
    df.rename(columns=rename_map, inplace=True)

    # Ensure numeric types where expected
    numeric_cols = ['NameIsFemale', 'NameFemininity', 'Fatalities', 'PropertyDamage', 'WindSpeed', 'MinPressure', 'Year']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Create log-transformed property damage (natural log). Use a floor at 1 to avoid -inf for zero damage.
    if 'PropertyDamage' in df.columns:
        df['LogPropertyDamage'] = np.log(df['PropertyDamage'].clip(lower=1.0))

    # Standardize continuous covariates (z-score) to aid interpretation and numerical stability
    for cont in ['NameFemininity', 'WindSpeed', 'MinPressure']:
        if cont in df.columns:
            mean = df[cont].mean(skipna=True)
            std = df[cont].std(skipna=True)
            if std == 0 or np.isnan(std):
                # If constant (unlikely), produce zeros to avoid division by zero
                df[cont + '_z'] = 0.0
            else:
                df[cont + '_z'] = (df[cont] - mean) / std

    # Center Year
    if 'Year' in df.columns:
        df['YearCentered'] = df['Year'] - df['Year'].mean()

    # Categorical conversions
    if 'Category' in df.columns:
        # Some category values may be numeric (1-5) but we want categorical dummies in modeling
        df['Category'] = df['Category'].astype('category')
    if 'Source' in df.columns:
        df['Source'] = df['Source'].astype('category')

    # Drop rows missing key variables for the primary analysis: Fatalities, name/gender, name femininity, and core covariates
    required_for_primary = ['Fatalities', 'NameIsFemale', 'NameFemininity_z', 'WindSpeed_z', 'MinPressure_z', 'YearCentered']
    existing_required = [c for c in required_for_primary if c in df.columns]
    if existing_required:
        df = df.dropna(subset=existing_required).reset_index(drop=True)

    # Final expected columns (these are the columns referenced in the modeling code and conceptual variables)
    # If any are missing, they will simply not be used by the model function, but we try to keep them present.
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit statistical models to test whether hurricanes with more feminine names are associated with less precautionary outcomes.

    Primary model: Negative binomial regression predicting Fatalities (count, overdispersed) with NameIsFemale and NameFemininity_z as key predictors, controlling for objective storm severity and year/source.

    Robustness / secondary model: OLS regression on log-transformed property damage (LogPropertyDamage) to test whether feminine names are associated with lower damage (another proxy for fewer precautions). Robust standard errors (HC3) are used for OLS.

    Returns a dictionary with fitted model result objects.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    results = {}

    # Build formula string for covariates used in both models
    # Use categorical indicators for Category and Source via C(...)
    covariates = 'NameIsFemale + NameFemininity_z + WindSpeed_z + MinPressure_z + YearCentered + C(Category) + C(Source)'

    # 1) Negative binomial for Fatalities (primary test)
    if 'Fatalities' in df.columns:
        nb_formula = f'Fatalities ~ {covariates}'
        try:
            nb_model = smf.glm(nb_formula, data=df, family=sm.families.NegativeBinomial()).fit()
            results['nb_fatalities'] = nb_model
        except Exception as e:
            # If the NB fit fails, return the exception text for diagnostics
            results['nb_fatalities_error'] = str(e)

    # 2) OLS on log property damage (robustness / secondary outcome)
    if 'LogPropertyDamage' in df.columns:
        ols_formula = f'LogPropertyDamage ~ {covariates}'
        try:
            ols_model = smf.ols(ols_formula, data=df).fit(cov_type='HC3')
            results['ols_log_property_damage'] = ols_model
        except Exception as e:
            results['ols_log_property_damage_error'] = str(e)

    # Return fitted model objects (or error messages). Caller can inspect model.summary() for details.
    return results


