from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/hurricane/add_features_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataset into a modeling-ready dataframe.

    Creates:
    - log_alldeaths: log(1 + alldeaths) (dependent variable)
    - log_ndam15: log(1 + ndam15) (control)
    - masfem_z: z-scored masfem (main independent variable)
    - year_c: year centered around its mean (control)
    - gender_female: integer binary copy of gender_mf (0=male name, 1=female name) for robustness

    Drops rows missing any of the core variables used in the primary specification.
    """
    df = df.copy()

    # Ensure relevant columns are numeric where expected
    numeric_cols = ['masfem', 'gender_mf', 'wind', 'category', 'min', 'alldeaths', 'ndam15', 'year']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing core variables required for primary model
    required = ['alldeaths', 'masfem', 'wind', 'category', 'min', 'year']
    df = df.dropna(subset=[c for c in required if c in df.columns])

    # Dependent variable: log transform to reduce skew
    df['log_alldeaths'] = np.log1p(df['alldeaths'].fillna(0))

    # Control: log of damage (ndam15) — fill missing as 0 then log1p
    if 'ndam15' in df.columns:
        df['log_ndam15'] = np.log1p(df['ndam15'].fillna(0))
    else:
        df['log_ndam15'] = 0.0

    # Independent variable: standardized masfem (z-score)
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1.0)

    # Year centered to aid interpretation of intercept and reduce collinearity
    df['year_c'] = df['year'] - df['year'].mean()

    # Binary female name indicator (robustness check)
    if 'gender_mf' in df.columns:
        df['gender_female'] = df['gender_mf'].astype(pd.Int64Dtype()).fillna(0).astype(int)
    else:
        df['gender_female'] = 0

    # Keep only columns needed for modeling to avoid accidental use of other columns
    keep_cols = [
        'log_alldeaths', 'masfem_z', 'gender_female', 'wind', 'category', 'min', 'log_ndam15', 'year_c'
    ]
    # Some columns may not exist depending on input; keep intersection
    keep_cols = [c for c in keep_cols if c in df.columns]

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit OLS models to test whether more feminine hurricane names are associated with larger fatalities (consistent with fewer precautions).

    Primary specification:
      log_alldeaths ~ masfem_z + wind + category + min + log_ndam15 + year_c

    Robustness specification (binary name gender):
      log_alldeaths ~ gender_female + wind + category + min + log_ndam15 + year_c

    Both models use heteroskedasticity-robust (HC3) standard errors.

    Returns a dictionary with keys 'model_masfem' and 'model_gender' containing the fitted OLS results objects.
    """
    import statsmodels.api as sm

    df = df.copy()

    # Ensure there are no NA in model columns (models expect numeric arrays without NA)
    model_cols_1 = ['masfem_z', 'wind', 'category', 'min', 'log_ndam15', 'year_c']
    model_cols_1 = [c for c in model_cols_1 if c in df.columns]
    model_cols_2 = ['gender_female', 'wind', 'category', 'min', 'log_ndam15', 'year_c']
    model_cols_2 = [c for c in model_cols_2 if c in df.columns]

    # Primary model (continuous femininity)
    X1 = df[model_cols_1].astype(float)
    X1 = sm.add_constant(X1)
    y = df['log_alldeaths'].astype(float)
    # Align indices and drop missing rows
    df1 = pd.concat([y, X1], axis=1).dropna()
    y1 = df1['log_alldeaths']
    X1 = df1.drop(columns=['log_alldeaths'])
    model1 = sm.OLS(y1, X1).fit(cov_type='HC3')

    # Robustness model (binary gender)
    X2 = df[model_cols_2].astype(float)
    X2 = sm.add_constant(X2)
    df2 = pd.concat([y, X2], axis=1).dropna()
    y2 = df2['log_alldeaths']
    X2 = df2.drop(columns=['log_alldeaths'])
    model2 = sm.OLS(y2, X2).fit(cov_type='HC3')

    # Return fitted models for inspection (summary, coefficients, standard errors, etc.)
    return {
        'model_masfem': model1,
        'model_gender': model2
    }


