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
    # Make a copy to avoid side effects
    df = df.copy()

    # Ensure numeric types where expected
    for col in ['masfem', 'gender_mf', 'alldeaths', 'ndam15', 'wind', 'category', 'min', 'elapsedyrs']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Keep rows with the variables we need for the primary analysis
    # We require a masfem score, a death count, and basic storm intensity controls
    required_cols = ['masfem', 'alldeaths', 'wind', 'category', 'min', 'elapsedyrs', 'source']
    df = df.dropna(subset=required_cols)

    # Standardize masfem for interpretable coefficients
    masfem_mean = df['masfem'].mean()
    masfem_std = df['masfem'].std(ddof=0)
    # protect against zero std
    if masfem_std == 0 or np.isnan(masfem_std):
        df['masfem_z'] = 0.0
    else:
        df['masfem_z'] = (df['masfem'] - masfem_mean) / masfem_std

    # Create alternate IV available in the dataset (binary female name)
    # Ensure it's numeric 0/1
    if 'gender_mf' in df.columns:
        df['gender_mf'] = df['gender_mf'].astype(int)

    # Log-transform damage for robustness checks (skewed, many large values)
    # Use log1p to handle zeros
    if 'ndam15' in df.columns:
        df['ndam15'] = pd.to_numeric(df['ndam15'], errors='coerce')
        df['log_ndam15'] = np.log1p(df['ndam15'].fillna(0))

    # Make sure alldeaths is integer non-negative
    df['alldeaths'] = df['alldeaths'].fillna(0).astype(int)

    # Keep only rows with non-negative integer deaths
    df = df[df['alldeaths'] >= 0]

    # If source is not categorical, keep as-is (will be dummy-coded in modeling)
    df['source'] = df['source'].astype(str)

    # Final: reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    # Build a model predicting counts of deaths (alldeaths) from masfem (standardized),
    # controlling for storm intensity and reporting/source differences.
    # Primary model: Negative Binomial GLM for count outcome (alldeaths).

    # Prepare exogenous variables (controls + IV). We'll dummy-code 'source' and include main predictors.
    exog_vars = ['masfem_z', 'wind', 'category', 'min', 'elapsedyrs']
    exog = df[exog_vars].copy()

    # Dummy-code source and add to exog (drop first to avoid multicollinearity)
    source_dummies = pd.get_dummies(df['source'].astype(str), prefix='source', drop_first=True)
    if not source_dummies.empty:
        exog = pd.concat([exog, source_dummies], axis=1)

    # Add constant
    exog = sm.add_constant(exog, has_constant='add')

    # Dependent variable
    endog = df['alldeaths']

    # Fit Negative Binomial GLM (log link is default). This is appropriate for overdispersed counts.
    try:
        nb_model = sm.GLM(endog, exog, family=sm.families.NegativeBinomial()).fit()
    except Exception:
        # Fallback to Poisson if NB fails to converge
        nb_model = sm.GLM(endog, exog, family=sm.families.Poisson()).fit()

    # Robustness check 1: Use binary gender coding as IV instead of continuous masfem
    if 'gender_mf' in df.columns:
        exog_bin = exog.copy()
        # replace standardized masfem with gender_mf
        exog_bin['masfem_z'] = df['gender_mf'].astype(float).values
        try:
            nb_model_gender = sm.GLM(endog, exog_bin, family=sm.families.NegativeBinomial()).fit()
        except Exception:
            nb_model_gender = sm.GLM(endog, exog_bin, family=sm.families.Poisson()).fit()
    else:
        nb_model_gender = None

    # Robustness check 2: Economic damages (log-transformed) predicted by masfem_z using OLS
    if 'log_ndam15' in df.columns and not df['log_ndam15'].isna().all():
        endog_dam = df['log_ndam15']
        # Use same exogenous structure (masfem_z + controls + source dummies)
        ols_model = sm.OLS(endog_dam, exog).fit()
    else:
        ols_model = None

    results = {
        'nb_model': nb_model,
        'nb_model_gender_binary_iv': nb_model_gender,
        'ols_log_damage_robustness': ols_model
    }

    return results


