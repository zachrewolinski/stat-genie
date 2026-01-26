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
    Transform the original dataset into a modeling-ready dataframe.

    Outputs (columns used in modeling):
      - Fatalities: original feature8 (numeric)
      - FemininityIndex: original feature4 (numeric)
      - FemininityIndex_c: centered FemininityIndex (used in model)
      - FemaleNameBinary: original feature6 coerced to 0/1
      - MTurkFemIndex: feature12 (kept for potential diagnostics)
      - MaxWind, MinPressure, Year: raw physical/time controls
      - MaxWind_z, MinPressure_z: standardized physical controls
      - Year_c: centered year
      - Cat_2..Cat_5: category dummy controls (category 1 is baseline)
      - LogFatalities: for OLS robustness (log(1+Fatalities))

    The function performs safe conversions and drops rows with missing key values.
    """
    df = df.copy()

    # ---- Rename columns to meaningful names ----
    rename_map = {
        'feature1': 'StormID',
        'feature2': 'Year',
        'feature3': 'StormName',
        'feature4': 'FemininityIndex',
        'feature5': 'MinPressure',
        'feature6': 'FemaleNameBinary',
        'feature7': 'Category',
        'feature8': 'Fatalities',
        'feature9': 'Damage2013',
        'feature10': 'YearsSince',
        'feature11': 'Source',
        'feature12': 'MTurkFemIndex',
        'feature13': 'MaxWind',
        'feature14': 'Damage2015'
    }
    df = df.rename(columns=rename_map)

    # Ensure numeric columns are numeric where expected
    numeric_cols = ['FemininityIndex', 'MinPressure', 'FemaleNameBinary', 'Category', 'Fatalities', 'Damage2013', 'YearsSince', 'MTurkFemIndex', 'MaxWind', 'Damage2015', 'Year']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing the main predictor or outcome
    df = df.dropna(subset=['FemininityIndex', 'Fatalities'])

    # Coerce FemaleNameBinary to strict 0/1 integer (if it contains 0/1 already)
    if 'FemaleNameBinary' in df.columns:
        # allow treat non-binary/missing as NA
        df['FemaleNameBinary'] = df['FemaleNameBinary'].apply(lambda x: int(x) if pd.notnull(x) else np.nan)
        df = df.dropna(subset=['FemaleNameBinary'])

    # Create centered femininity index used in the model
    df['FemininityIndex_c'] = df['FemininityIndex'] - df['FemininityIndex'].mean()

    # Create outcome transforms for modeling and robustness checks
    df['LogFatalities'] = np.log1p(df['Fatalities'])

    # Center year to improve interpretability and numerical stability
    if 'Year' in df.columns:
        df['Year_c'] = df['Year'] - df['Year'].mean()
    else:
        df['Year_c'] = 0.0

    # Standardize physical severity measures (z-scores). Use population sd (ddof=0) to be consistent.
    if 'MaxWind' in df.columns:
        mw_mean = df['MaxWind'].mean()
        mw_std = df['MaxWind'].std(ddof=0)
        if mw_std == 0 or np.isnan(mw_std):
            df['MaxWind_z'] = 0.0
        else:
            df['MaxWind_z'] = (df['MaxWind'] - mw_mean) / mw_std
    else:
        df['MaxWind_z'] = 0.0

    if 'MinPressure' in df.columns:
        mp_mean = df['MinPressure'].mean()
        mp_std = df['MinPressure'].std(ddof=0)
        if mp_std == 0 or np.isnan(mp_std):
            df['MinPressure_z'] = 0.0
        else:
            df['MinPressure_z'] = (df['MinPressure'] - mp_mean) / mp_std
    else:
        df['MinPressure_z'] = 0.0

    # Create explicit category dummy controls for Saffir-Simpson categories 2-5 (1 is baseline)
    # Ensure Category is integer-like
    df['Category'] = pd.to_numeric(df['Category'], errors='coerce').astype('Int64')
    for lvl in [2, 3, 4, 5]:
        col = f'Cat_{lvl}'
        df[col] = (df['Category'] == lvl).astype(int)

    # Final safety drop: keep only rows with non-missing values in columns we will use in modeling
    required_for_model = ['Fatalities', 'FemininityIndex_c', 'FemaleNameBinary', 'MaxWind_z', 'MinPressure_z', 'Year_c', 'Cat_2', 'Cat_3', 'Cat_4', 'Cat_5']
    required_available = [c for c in required_for_model if c in df.columns]
    df = df.dropna(subset=['Fatalities', 'FemininityIndex_c', 'FemaleNameBinary'])

    # Return the transformed dataframe with all created columns retained
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit the main Negative Binomial GLM of Fatalities on femininity of name and controls.

    Model specification (primary):
      Fatalities ~ FemininityIndex_c + FemaleNameBinary + MaxWind_z + MinPressure_z + Year_c + Cat_2 + Cat_3 + Cat_4 + Cat_5

    Estimation details:
      - Family: Negative Binomial (appropriate for count data with overdispersion)
      - We also compute robust (HC3) standard errors for inference
      - Robustness: OLS on LogFatalities with HC3 SEs

    Returns a dictionary with the fitted models (statsmodels results objects):
      {'nb_model': <GLM Results>, 'nb_robust': <GLM Results with robust cov>, 'ols_logfatalities': <OLS Results with robust cov>}
    """
    import statsmodels.api as sm

    df = df.copy()

    # Define outcome and predictors exactly as in the transform step
    y = df['Fatalities']

    # Predictor list: the centered femininity measure and the female name binary indicator
    predictors = ['FemininityIndex_c', 'FemaleNameBinary']

    # Controls: standardized physical measures, centered year, and category dummies
    control_vars = []
    for v in ['MaxWind_z', 'MinPressure_z', 'Year_c', 'Cat_2', 'Cat_3', 'Cat_4', 'Cat_5']:
        if v in df.columns:
            control_vars.append(v)

    X = df[predictors + control_vars]
    X = sm.add_constant(X, has_constant='add')

    # Fit Negative Binomial GLM (counts, allows overdispersion)
    try:
        nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit()
    except Exception as e:
        # If NB fails to converge, fall back to Poisson with robust SEs (less ideal)
        nb_model = sm.GLM(y, X, family=sm.families.Poisson()).fit()

    # Robust covariance (HC3) for inference
    try:
        nb_robust = nb_model.get_robustcov_results(cov_type='HC3')
    except Exception:
        nb_robust = nb_model

    # Robustness check: OLS on log(1+Fatalities) with HC3 SEs
    y_log = df['LogFatalities']
    ols_model = sm.OLS(y_log, X).fit(cov_type='HC3')

    # Return results; caller can inspect summaries, coef/pvalues, etc.
    return {
        'nb_model': nb_model,
        'nb_robust': nb_robust,
        'ols_logfatalities': ols_model
    }


