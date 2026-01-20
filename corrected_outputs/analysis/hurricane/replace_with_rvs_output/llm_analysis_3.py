from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/hurricane/replace_with_rvs_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy to avoid modifying original
    df = df.copy()

    # Keep only columns we will use and drop rows with missing key variables
    required_cols = [
        'masfem', 'gender_mf', 'alldeaths', 'ndam15', 'wind', 'category',
        'min', 'elapsedyrs', 'source', 'year', 'masfem_mturk'
    ]
    # Drop rows missing core information needed for models
    df = df.dropna(subset=required_cols)

    # Ensure numeric dtypes for key numeric columns
    numeric_cols = ['masfem', 'alldeaths', 'ndam15', 'wind', 'category', 'min', 'elapsedyrs', 'year', 'masfem_mturk']
    for col in numeric_cols:
        # coerce errors -> NaN then drop above handled missing values
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Re-drop rows that became NaN after coercion
    df = df.dropna(subset=numeric_cols)

    # Create log-transformed outcome variables (handle zeros with log1p)
    df['log_alldeaths'] = np.log1p(df['alldeaths'].astype(float))
    df['log_ndam15'] = np.log1p(df['ndam15'].astype(float))

    # Standardize/center the masfem ratings (mean 0, sd 1) for easier interpretation
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1.0)

    # Also standardize mturk rating for robustness checks
    df['masfem_mturk_z'] = (df['masfem_mturk'] - df['masfem_mturk'].mean()) / (df['masfem_mturk'].std(ddof=0) if df['masfem_mturk'].std(ddof=0) != 0 else 1.0)

    # Ensure category is categorical for modeling
    df['category'] = df['category'].astype('category')

    # Ensure source is treated as a categorical label (do not one-hot here; will use categorical coding in formula)
    df['source'] = df['source'].astype(str).fillna('unknown')

    # Ensure gender_mf is integer 0/1
    df['gender_mf'] = df['gender_mf'].astype(int)

    # Final check: drop rows with any remaining NA in model variables
    model_vars = [
        'log_alldeaths', 'log_ndam15', 'masfem_z', 'gender_mf', 'wind', 'category',
        'min', 'elapsedyrs', 'year', 'source', 'masfem_mturk_z'
    ]
    df = df.dropna(subset=model_vars)

    # Reset index
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """Run primary regression analyses.

    Returns a dictionary with two fitted models (death and damage) as statsmodels objects.
    Both models estimate the association between name femininity (masfem_z) and the outcome,
    controlling for storm intensity and temporal/source covariates. Robust (HC3) standard errors
    are used.
    """
    import statsmodels.formula.api as smf

    # Primary formula: log deaths
    formula_deaths = (
        'log_alldeaths ~ masfem_z + gender_mf + wind + C(category) + min + elapsedyrs + year + C(source)'
    )

    # Primary formula: log damage (2015-adjusted)
    formula_damage = (
        'log_ndam15 ~ masfem_z + gender_mf + wind + C(category) + min + elapsedyrs + year + C(source)'
    )

    # Fit OLS with heteroskedasticity-robust standard errors (HC3)
    death_model = smf.ols(formula_deaths, data=df).fit(cov_type='HC3')
    damage_model = smf.ols(formula_damage, data=df).fit(cov_type='HC3')

    # Robustness: replace masfem_z with masfem_mturk_z (alternative IV) for deaths
    formula_deaths_mturk = formula_deaths.replace('masfem_z', 'masfem_mturk_z')
    death_model_mturk = smf.ols(formula_deaths_mturk, data=df).fit(cov_type='HC3')

    results = {
        'death_model': death_model,
        'damage_model': damage_model,
        'death_model_mturk_iv': death_model_mturk
    }

    return results


