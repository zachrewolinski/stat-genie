from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/fish/add_features_output/fish.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw fishing dataset to the modeling-ready dataframe.

    Outputs (columns added/guaranteed):
    - fish_per_hour: fish_caught / hours (float)
    - total_persons: persons + child (int)
    - log_hours: natural log of hours (float) to be used as an offset in count models
    - county: string (no-missing, filled with 'Unknown')
    - livebait, camper: coerced to int (0/1)
    - age: numeric (may contain NaN if not provided)
    """
    df = df.copy()

    # Ensure required numeric columns exist and coerce
    df['fish_caught'] = pd.to_numeric(df['fish_caught'], errors='coerce')
    df['hours'] = pd.to_numeric(df['hours'], errors='coerce')

    # Drop rows missing fish count or hours (cannot compute a rate without hours)
    df = df.dropna(subset=['fish_caught', 'hours'])

    # Remove non-positive (zero or negative) hours because they cannot be used as exposure
    df = df[df['hours'] > 0].copy()

    # Coerce binary predictors to integer 0/1 (fill missing as 0 when reasonable)
    if 'livebait' in df.columns:
        df['livebait'] = df['livebait'].fillna(0).astype(int)
    else:
        df['livebait'] = 0

    if 'camper' in df.columns:
        df['camper'] = df['camper'].fillna(0).astype(int)
    else:
        df['camper'] = 0

    # Persons and children: fill missing conservatively and compute total persons
    if 'persons' in df.columns:
        df['persons'] = pd.to_numeric(df['persons'], errors='coerce').fillna(1).astype(int)
    else:
        df['persons'] = 1

    if 'child' in df.columns:
        df['child'] = pd.to_numeric(df['child'], errors='coerce').fillna(0).astype(int)
    else:
        df['child'] = 0

    df['total_persons'] = df['persons'] + df['child']

    # Age: coerce to numeric (may be NaN)
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
    else:
        df['age'] = np.nan

    # Exposure (log of hours) for count models and a simple rate column
    df['log_hours'] = np.log(df['hours'])
    df['fish_per_hour'] = df['fish_caught'] / df['hours']

    # County: ensure categorical string, fill missing
    if 'county' in df.columns:
        df['county'] = df['county'].astype(str).fillna('Unknown')
    else:
        df['county'] = 'Unknown'

    # Final: keep columns needed for modeling and inspection
    # (we keep original columns plus derived columns)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit count models for fish_caught using park-hours as exposure (offset).

    Procedure:
    1. Fit a Poisson GLM with offset = log_hours to model counts as a rate (fish per hour).
    2. Compute dispersion statistic (Pearson chi2 / df_resid). If there is notable overdispersion (>1.5), try a Negative Binomial GLM.

    Returns a dictionary with:
    - 'poisson': fitted Poisson results (statsmodels results object)
    - 'overdispersion': computed dispersion value
    - 'negative_binomial': fitted NB results (if fitted, else None)
    - if negative binomial fitting fails, an error message may be provided under 'negative_binomial_error'
    """
    import statsmodels.formula.api as smf

    # Copy to avoid modifying original
    df = df.copy()

    # Formula: model fish_caught as function of main predictors and county fixed effects
    # Exposure (hours) is represented by offset=log_hours so the model estimates fish-per-hour
    formula = 'fish_caught ~ livebait + camper + total_persons + child + age + C(county)'

    # Fit Poisson with offset
    poisson_model = smf.glm(formula=formula, data=df, family=sm.families.Poisson(), offset=df['log_hours']).fit()

    # Compute dispersion: Pearson chi2 / df_resid
    try:
        pearson_chi2 = poisson_model.pearson_chi2
        df_resid = poisson_model.df_resid
        overdispersion = pearson_chi2 / df_resid if df_resid > 0 else np.nan
    except Exception:
        # If pearson_chi2 not available for some reason
        overdispersion = np.nan

    results = {
        'poisson': poisson_model,
        'overdispersion': overdispersion,
        'negative_binomial': None
    }

    # If overdispersion detected, attempt a Negative Binomial GLM
    if not np.isnan(overdispersion) and overdispersion > 1.5:
        try:
            # Try statsmodels GLM NegativeBinomial family (available in recent statsmodels)
            nb_model = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial(), offset=df['log_hours']).fit()
            results['negative_binomial'] = nb_model
        except Exception as e_nb:
            # Fallback attempt using the discrete NegativeBinomial (different API). Build a design matrix.
            try:
                from statsmodels.discrete.discrete_model import NegativeBinomial as NBdiscrete
                # Build design matrix using patsy via formula but without the offset; include dummies for county
                # We'll construct exog explicitly (including intercept)
                # Use pandas get_dummies on the same predictors in the formula
                exog_vars = ['livebait', 'camper', 'total_persons', 'child', 'age']
                exog = pd.get_dummies(df[exog_vars + ['county']], drop_first=True)
                exog = sm.add_constant(exog, has_constant='add')
                endog = df['fish_caught']
                nb2 = NBdiscrete(endog, exog).fit(disp=False)
                results['negative_binomial'] = nb2
            except Exception as e_fallback:
                results['negative_binomial_error'] = f"NB fit failed: {str(e_nb)} | fallback error: {str(e_fallback)}"

    return results


