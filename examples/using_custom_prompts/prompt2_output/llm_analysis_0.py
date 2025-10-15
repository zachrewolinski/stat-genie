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
    Perform all data cleaning and create the final columns required for modeling.

    Produces:
      - log_alldeaths : np.log1p(alldeaths)
      - masfem_z      : z-scored masfem (higher = more feminine)
      - wind_z        : z-scored wind
      - min_z         : z-scored min (pressure)
      - log_ndam15_z  : z-scored log1p(ndam15)
      - category      : kept as original integer (1-5)
      - year_centered : year - mean(year)
      - elapsedyrs    : filled/missing handled

    Returns a dataframe with only the modeling columns.
    """
    df = df.copy()

    # Ensure required numeric columns are present; coerce if necessary
    for col in ['alldeaths', 'masfem', 'wind', 'min', 'ndam15', 'category', 'year']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing the key outcome or the main IV
    df = df.dropna(subset=['alldeaths', 'masfem'])

    # Create logged outcome
    df['log_alldeaths'] = np.log1p(df['alldeaths'])

    # Create logged damage (2015-adjusted damages) if available
    if 'ndam15' in df.columns:
        df['log_ndam15'] = np.log1p(df['ndam15'])
    else:
        # If ndam15 not present, create a placeholder column of zeros to avoid errors
        df['log_ndam15'] = 0.0

    # Fill elapsedyrs with median if present, otherwise create 0s
    if 'elapsedyrs' in df.columns:
        df['elapsedyrs'] = pd.to_numeric(df['elapsedyrs'], errors='coerce')
        df['elapsedyrs'] = df['elapsedyrs'].fillna(df['elapsedyrs'].median())
    else:
        df['elapsedyrs'] = 0.0

    # Standardize (z-score) continuous predictors to make coefficients comparable
    # Use population std (ddof=0) for stability in small samples
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1)
    df['wind_z'] = (df['wind'] - df['wind'].mean()) / (df['wind'].std(ddof=0) if df['wind'].std(ddof=0) != 0 else 1)
    df['min_z'] = (df['min'] - df['min'].mean()) / (df['min'].std(ddof=0) if df['min'].std(ddof=0) != 0 else 1)
    df['log_ndam15_z'] = (df['log_ndam15'] - df['log_ndam15'].mean()) / (df['log_ndam15'].std(ddof=0) if df['log_ndam15'].std(ddof=0) != 0 else 1)

    # Center year to reduce collinearity and ease interpretation
    df['year_centered'] = df['year'] - df['year'].mean()

    # Ensure category is integer and drop rows without category
    if 'category' in df.columns:
        df['category'] = pd.to_numeric(df['category'], errors='coerce')
        df = df.dropna(subset=['category'])
        # Keep as-is (1-5). Optionally one-hot encode in modeling stage.
        df['category'] = df['category'].astype(int)
    else:
        # If category missing, create a default category 1
        df['category'] = 1

    # Final selection of columns needed for the model
    final_cols = ['log_alldeaths', 'masfem_z', 'wind_z', 'min_z', 'category', 'log_ndam15_z', 'year_centered', 'elapsedyrs']

    # Return cleaned dataframe with only the necessary columns (copy to avoid view issues)
    return df[final_cols].copy()


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> float:
    """
    Fit an OLS regression predicting log fatalities from femininity of the hurricane name
    controlling for storm intensity, damages, year, and elapsed years. Returns the
    coefficient on masfem_z (standardized femininity index). Uses robust (HC3) standard
    errors when fitting but returns the point estimate as the single-number summary.

    Model: log_alldeaths ~ masfem_z + wind_z + min_z + category + log_ndam15_z + year_centered + elapsedyrs
    """
    import statsmodels.api as sm

    df = df.copy()

    # Ensure required columns are present
    required = ['log_alldeaths', 'masfem_z', 'wind_z', 'min_z', 'category', 'log_ndam15_z', 'year_centered', 'elapsedyrs']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Design matrix
    X = df[['masfem_z', 'wind_z', 'min_z', 'category', 'log_ndam15_z', 'year_centered', 'elapsedyrs']]
    X = sm.add_constant(X)
    y = df['log_alldeaths']

    # Fit OLS with robust covariance (HC3) to reduce sensitivity to heteroskedasticity
    model = sm.OLS(y, X).fit(cov_type='HC3')

    # Extract the coefficient on masfem_z as the key summary number
    coef_masfem = model.params.get('masfem_z')

    # Return as Python float
    return float(coef_masfem)


