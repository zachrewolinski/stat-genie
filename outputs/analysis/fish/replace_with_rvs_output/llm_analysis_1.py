from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/fish/replace_with_rvs_output/fish.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Drop rows with missing key measurement or predictor values
    df = df.dropna(subset=['fish_caught', 'hours', 'livebait', 'camper', 'persons', 'child'])

    # Ensure numeric types
    df['fish_caught'] = pd.to_numeric(df['fish_caught'], errors='coerce')
    df['hours'] = pd.to_numeric(df['hours'], errors='coerce')
    df['livebait'] = pd.to_numeric(df['livebait'], errors='coerce').astype(int)
    df['camper'] = pd.to_numeric(df['camper'], errors='coerce').astype(int)
    df['persons'] = pd.to_numeric(df['persons'], errors='coerce')
    df['child'] = pd.to_numeric(df['child'], errors='coerce')

    # Remove any rows with invalid or zero hours (cannot compute rate or log offset)
    df = df.dropna(subset=['fish_caught', 'hours', 'persons', 'child'])
    df = df[df['hours'] > 0]

    # Construct group-size variables
    df['total_people'] = df['persons'] + df['child']
    # If for some reason total_people is zero (shouldn't be given persons >=1), avoid division by zero
    df['prop_child'] = df['child'] / df['total_people']
    df['prop_child'] = df['prop_child'].fillna(0)

    # Descriptive rate outcome and log-hours for exposure offset
    df['fish_per_hour'] = df['fish_caught'] / df['hours']
    df['log_hours'] = np.log(df['hours'])

    # Keep only columns necessary for modeling + diagnostics
    # (This is optional — returning the full DF with added columns is useful for checks)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # Build model to explain fish counts accounting for differing hours (exposure)
    # We'll use a Negative Binomial GLM with an offset = log(hours) to model count / hour (i.e., rate).

    df = df.copy()

    # Predictor set as described in conceptual variables
    predictors = ['livebait', 'camper', 'total_people', 'prop_child']

    # Ensure predictors present
    missing = [p for p in predictors if p not in df.columns]
    if missing:
        raise ValueError(f"Missing predictors for modeling: {missing}")

    X = df[predictors]
    X = sm.add_constant(X)
    y = df['fish_caught']
    offset = df['log_hours']

    # Fit Negative Binomial GLM with log-link and offset(log(hours))
    # This models: E[fish_caught] = hours * exp(X * beta)
    model_glm = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset)
    results = model_glm.fit()

    # Attach a small diagnostic summary (dispersion approximation)
    # Approximate Pearson chi2 / df to inspect overdispersion
    pearson_chi2 = ((results.resid_pearson**2).sum())
    df_resid = results.df_resid if hasattr(results, 'df_resid') else (len(df) - X.shape[1])
    dispersion = pearson_chi2 / df_resid if df_resid != 0 else np.nan

    # Package results: fitted model plus diagnostics and a simple table of marginal effects (exponentiated coefficients => multiplicative effects on rate)
    summary = {
        'results': results,
        'dispersion_approx': dispersion,
        'exp_coef': np.exp(results.params),
        'model_formula_predictors': predictors
    }

    return summary


