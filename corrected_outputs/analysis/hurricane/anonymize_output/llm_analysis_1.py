from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/hurricane/anonymize_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform original dataset (features named feature1..feature14) into a cleaned dataframe
    with interpretable column names and derived variables for modeling.

    Outputs (added/renamed columns used in the model):
    - MasFem: original continuous masculinity-femininity index (feature4)
    - MasFem_z: standardized MasFem (z-score)
    - IsFemaleName: binary indicator from feature6 (0 male, 1 female)
    - Category: Saffir-Simpson category (feature7)
    - Fatalities: number of deaths (feature8)
    - log_fatalities: np.log1p(Fatalities)
    - MaxWind: maximum wind speed at landfall (feature13)
    - MaxWind_z: standardized MaxWind
    - MinPressure: minimum central pressure at landfall (feature5)
    - MinPressure_z: standardized MinPressure
    - Year: year of event (feature2)
    - Year_c: centered Year
    - Source: data source (feature11) kept as categorical

    The function drops rows with missing values in the variables required for the main model.
    """
    # work on a copy
    df = df.copy()

    # Rename columns to meaningful names
    rename_map = {
        'feature1': 'ID',
        'feature2': 'Year',
        'feature3': 'Name',
        'feature4': 'MasFem',        # continuous masculinity-femininity index
        'feature5': 'MinPressure',   # minimum pressure at landfall
        'feature6': 'IsFemaleName',  # binary gender indicator (0 male, 1 female)
        'feature7': 'Category',      # Saffir-Simpson category (1-5)
        'feature8': 'Fatalities',    # total deaths
        'feature9': 'Damage2013',    # damage normalized to 2013
        'feature10': 'YearsSince',
        'feature11': 'Source',
        'feature12': 'MTurkMasFem',
        'feature13': 'MaxWind',      # max wind speed at landfall
        'feature14': 'Damage2015'
    }
    df = df.rename(columns=rename_map)

    # Keep only the columns we will use
    required_cols = ['MasFem', 'IsFemaleName', 'Category', 'Fatalities', 'MaxWind', 'MinPressure', 'Year', 'Source']

    # Drop rows missing any of the required columns
    df = df.dropna(subset=required_cols)

    # Ensure numeric types where expected
    numeric_cols = ['MasFem', 'IsFemaleName', 'Category', 'Fatalities', 'MaxWind', 'MinPressure', 'Year']
    for c in numeric_cols:
        # coerce errors to NaN and then drop above (already dropped), but ensure dtype
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Derived variables
    # log(1 + fatalities) to reduce skew and handle zeros
    df['log_fatalities'] = np.log1p(df['Fatalities'].astype(float))

    # Standardize continuous predictors for interpretability
    df['MasFem_z'] = (df['MasFem'] - df['MasFem'].mean()) / (df['MasFem'].std(ddof=0) if df['MasFem'].std(ddof=0) != 0 else 1.0)
    df['MaxWind_z'] = (df['MaxWind'] - df['MaxWind'].mean()) / (df['MaxWind'].std(ddof=0) if df['MaxWind'].std(ddof=0) != 0 else 1.0)
    df['MinPressure_z'] = (df['MinPressure'] - df['MinPressure'].mean()) / (df['MinPressure'].std(ddof=0) if df['MinPressure'].std(ddof=0) != 0 else 1.0)

    # Center year to improve interpretability and numerical stability
    df['Year_c'] = df['Year'] - df['Year'].mean()

    # Ensure categorical columns are of type object/category
    df['Category'] = df['Category'].astype('category')
    df['Source'] = df['Source'].astype('category')

    # Ensure IsFemaleName is integer 0/1
    df['IsFemaleName'] = df['IsFemaleName'].astype(int)

    # Final drop to remove any rows with newly introduced NA (should be rare)
    model_cols = ['log_fatalities', 'MasFem_z', 'IsFemaleName', 'MaxWind_z', 'MinPressure_z', 'Category', 'Year_c', 'Source']
    df = df.dropna(subset=model_cols)

    # Return transformed df (includes both original and derived columns)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Runs an OLS regression testing whether more feminine hurricane names are associated with
    higher fatalities after controlling for objective storm severity and temporal/source effects.

    Primary specification:
      log_fatalities ~ MasFem_z + IsFemaleName + MaxWind_z + MinPressure_z + C(Category) + Year_c + MasFem_z:MaxWind_z

    The interaction MasFem_z:MaxWind_z tests whether the femininity effect differs by storm strength
    (theoretically, if feminine names lead to fewer precautions, the effect of femininity on fatalities
    might be stronger for storms where precaution-taking matters more).

    Robust (HC3) standard errors are used because of small sample size and potential heteroskedasticity.
    """
    import statsmodels.formula.api as smf

    # Ensure the model columns exist in the dataframe provided
    required = ['log_fatalities', 'MasFem_z', 'IsFemaleName', 'MaxWind_z', 'MinPressure_z', 'Category', 'Year_c', 'Source']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Dataframe is missing required columns for modeling: {missing}")

    # Define formula: treat Category and Source as categorical (fixed effects)
    formula = 'log_fatalities ~ MasFem_z + IsFemaleName + MaxWind_z + MinPressure_z + C(Category) + Year_c + MasFem_z:MaxWind_z + C(Source)'

    # Fit OLS with robust (HC3) standard errors
    model = smf.ols(formula=formula, data=df)
    results = model.fit(cov_type='HC3')

    # Return the fitted results object (caller can inspect summary, params, etc.)
    return results


