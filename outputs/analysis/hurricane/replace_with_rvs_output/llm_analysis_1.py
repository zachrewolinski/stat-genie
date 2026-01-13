from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/replace_with_rvs_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataframe to the analysis-ready dataframe.

    Produces/ensures the following columns used by the model:
      - masfem_scaled: standardized masfem (continuous IV)
      - gender_female: binary indicator (0/1) copied from gender_mf
      - death_count: integer count of fatalities (alldeaths)
      - log_alldeaths: log(alldeaths + 1) for descriptive checks / OLS models
      - log_ndam15: log(ndam15 + 1) (alternative outcome: economic damage)
      - min_pressure: renamed 'min' -> 'min_pressure'
      - wind, category, elapsedyrs, source preserved (and source as categorical)

    Drops rows missing the core variables required for the principal analyses.
    """
    df = df.copy()

    # Rename min -> min_pressure for clarity
    if 'min' in df.columns:
        df = df.rename(columns={'min': 'min_pressure'})

    # Core variables required for analysis
    core_vars = ['masfem', 'gender_mf', 'alldeaths', 'ndam15', 'wind', 'category', 'elapsedyrs', 'source']
    missing_core = [c for c in core_vars if c not in df.columns]
    if missing_core:
        raise KeyError(f"Missing required columns for transform: {missing_core}")

    # Drop rows with missing key variables (we need masfem and death/damage and key controls)
    df = df.dropna(subset=['masfem', 'alldeaths', 'ndam15', 'wind', 'category', 'elapsedyrs', 'source'])

    # Ensure numeric types where expected
    df['masfem'] = pd.to_numeric(df['masfem'], errors='coerce')
    df['alldeaths'] = pd.to_numeric(df['alldeaths'], errors='coerce').fillna(0).astype(int)
    df['ndam15'] = pd.to_numeric(df['ndam15'], errors='coerce').fillna(0)
    df['wind'] = pd.to_numeric(df['wind'], errors='coerce')
    # Convert category to a plain integer dtype (not pandas nullable Int64) so that patsy/statsmodels can handle it.
    df['category'] = pd.to_numeric(df['category'], errors='coerce').astype(int)
    df['elapsedyrs'] = pd.to_numeric(df['elapsedyrs'], errors='coerce')

    # Create analysis columns
    # Standardize masfem to mean 0, sd 1 (z-score)
    masfem_mean = df['masfem'].mean()
    masfem_std = df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1.0
    df['masfem_scaled'] = (df['masfem'] - masfem_mean) / masfem_std

    # Binary female name indicator
    # Original column 'gender_mf' is 0/1 (0=male, 1=female) per schema
    df['gender_female'] = df['gender_mf'].astype(int)

    # Dependent variable: death counts
    df['death_count'] = df['alldeaths'].astype(int)
    # log transform for alternative linear models / diagnostics
    df['log_alldeaths'] = np.log(df['alldeaths'] + 1)

    # Alternative outcome (economic damage) log-transformed
    df['log_ndam15'] = np.log(df['ndam15'] + 1)

    # Ensure min_pressure exists and numeric
    if 'min_pressure' in df.columns:
        df['min_pressure'] = pd.to_numeric(df['min_pressure'], errors='coerce')
    else:
        # if original 'min' was not present (unexpected), create an NA column
        df['min_pressure'] = np.nan

    # Make source categorical
    df['source'] = df['source'].astype('category')

    # Final drop: any remaining rows with NA in the analysis columns
    analysis_cols = ['masfem_scaled', 'gender_female', 'death_count', 'wind', 'category', 'min_pressure', 'elapsedyrs', 'source']
    df = df.dropna(subset=analysis_cols)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit statistical models to test whether more feminine hurricane names are associated with fewer precautions
    (proxied by higher fatalities) while controlling for storm intensity and other covariates.

    Models returned:
      - nb_deaths: Negative binomial regression of death_count on masfem_scaled (+ controls)
      - ols_damage: OLS regression of log_ndam15 on masfem_scaled (+ controls) with robust SE (HC3)

    Returns a dictionary of fitted model results objects.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    results = {}

    # Ensure the dataframe contains the expected analysis columns
    required = ['death_count', 'masfem_scaled', 'gender_female', 'wind', 'min_pressure', 'elapsedyrs', 'category', 'source', 'log_ndam15']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for modeling: {missing}")

    # 1) Negative binomial for fatalities (count outcome)
    # Formula: death_count ~ masfem_scaled + gender_female + wind + min_pressure + elapsedyrs + C(category) + C(source)
    nb_formula = 'death_count ~ masfem_scaled + gender_female + wind + min_pressure + elapsedyrs + C(category) + C(source)'
    try:
        nb_model = smf.glm(nb_formula, data=df, family=sm.families.NegativeBinomial()).fit()
    except Exception:
        # fallback to Poisson with robust SE if NegativeBinomial fails to converge
        poisson_model = smf.glm(nb_formula, data=df, family=sm.families.Poisson()).fit()
        nb_model = poisson_model

    results['nb_deaths'] = nb_model

    # 2) OLS on log economic damage (alternative dependent variable), robust SE
    ols_formula = 'log_ndam15 ~ masfem_scaled + gender_female + wind + min_pressure + elapsedyrs + C(category) + C(source)'
    ols_model = smf.ols(ols_formula, data=df).fit(cov_type='HC3')
    results['ols_damage'] = ols_model

    # Print brief summaries (useful when running interactively)
    try:
        print('\nNegative binomial / Poisson model for fatalities:')
        print(nb_model.summary())
        print('\nOLS (robust SE) for log damage:')
        print(ols_model.summary())
    except Exception:
        # If printing fails (e.g., non-interactive environment), ignore
        pass

    return results