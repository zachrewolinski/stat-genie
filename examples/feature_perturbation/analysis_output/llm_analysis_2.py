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
    Transform the raw input dataframe into a cleaned dataframe with the variables used in the model.

    Expected original columns (per schema):
      - feature13: total number of deaths caused by the hurricane (Fatalities)
      - feature9: masculinity-femininity index from independent coders (higher = more feminine)
      - feature12: binary gender indicator for name (0=male, 1=female)
      - feature7: maximum wind speed at landfall
      - feature4: Saffir-Simpson category
      - feature14: minimum pressure at landfall
      - feature5: year
      - feature8: property damage (2015-normalized)
      - feature6: storm name
      - feature2: storm id

    The function will:
      - rename columns to intuitive names
      - coerce numeric columns to numeric, drop rows with missing critical values
      - create LogFatalities = log(Fatalities + 1)
      - center the NameMasFem index (NameMasFem_c)
      - coerce FemaleName to integer
      - create LogPropertyDamage2015 = log(PropertyDamage2015 + 1)
    """
    df = df.copy()

    # Rename schema columns to meaningful names
    rename_map = {
        'feature13': 'Fatalities',
        'feature9': 'NameMasFemIndex',
        'feature12': 'FemaleName',
        'feature7': 'MaxWindSpeed',
        'feature4': 'Category',
        'feature14': 'MinPressure',
        'feature5': 'Year',
        'feature8': 'PropertyDamage2015',
        'feature6': 'StormName',
        'feature2': 'StormID'
    }
    df = df.rename(columns=rename_map)

    # Ensure numeric columns are numeric
    numeric_cols = ['Fatalities', 'NameMasFemIndex', 'FemaleName', 'MaxWindSpeed', 'Category', 'MinPressure', 'Year', 'PropertyDamage2015']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing the core variables needed for this analysis
    df = df.dropna(subset=['Fatalities', 'NameMasFemIndex', 'MaxWindSpeed', 'Category'])

    # Dependent variable: log-transform fatalities (handle zeros)
    df['LogFatalities'] = np.log(df['Fatalities'].astype(float) + 1)

    # Independent variable: center the masculinity-femininity index for interpretability
    df['NameMasFem_c'] = df['NameMasFemIndex'].astype(float) - df['NameMasFemIndex'].astype(float).mean()

    # Ensure binary female indicator is 0/1 integer
    df['FemaleName'] = df['FemaleName'].fillna(0).astype(int)

    # Control: logged property damage (2015-normalized) to stabilize skew
    if 'PropertyDamage2015' in df.columns:
        df['LogPropertyDamage2015'] = np.log(df['PropertyDamage2015'].astype(float).fillna(0) + 1)
    else:
        df['LogPropertyDamage2015'] = 0.0

    # Keep only columns that will be used in modeling (but keep metadata columns too)
    cols_needed = [
        'StormID', 'StormName', 'Fatalities', 'LogFatalities', 'NameMasFemIndex', 'NameMasFem_c', 'FemaleName',
        'MaxWindSpeed', 'Category', 'MinPressure', 'Year', 'PropertyDamage2015', 'LogPropertyDamage2015'
    ]
    # keep only those that exist
    cols_keep = [c for c in cols_needed if c in df.columns]
    df = df[cols_keep]

    # Final quick cleans: drop any rows where the DV or key IV is missing after transforms
    df = df.dropna(subset=['LogFatalities', 'NameMasFem_c'])

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS model predicting log(fatalities + 1) from the centered name femininity index
    and physical / temporal controls. Uses HC3 robust standard errors.

    Model specification:
      LogFatalities ~ NameMasFem_c + FemaleName + MaxWindSpeed + Category + MinPressure + Year + LogPropertyDamage2015

    Returns the fitted statsmodels results object.
    """
    # Ensure required columns are present
    required_cols = ['LogFatalities', 'NameMasFem_c', 'FemaleName', 'MaxWindSpeed', 'Category', 'MinPressure', 'Year', 'LogPropertyDamage2015']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Design matrix
    X = df[['NameMasFem_c', 'FemaleName', 'MaxWindSpeed', 'Category', 'MinPressure', 'Year', 'LogPropertyDamage2015']].astype(float)
    X = sm.add_constant(X, has_constant='add')
    y = df['LogFatalities'].astype(float)

    # Fit OLS with robust (HC3) standard errors
    ols_model = sm.OLS(y, X)
    results = ols_model.fit(cov_type='HC3')

    # Return the full results object so the caller can inspect summary, params, conf_int, etc.
    return results


