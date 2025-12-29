from typing import Any, Dict
import numpy as np
import pandas as pd
import statsmodels.api as sm


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and transform raw fishing trip dataframe into the final dataframe
    required by the analysis.

    Required output columns (do not change names):
      - 'livebait' (binary 0/1)
      - 'camper' (integer >= 0)
      - 'persons' (integer > 0)
      - 'child' (binary 0/1)
      - 'fish_caught' (numeric count)
      - 'hours' (numeric > 0)
      - 'log_hours' (log of hours)
      - 'fish_per_hour' (fish_caught / hours)
      - 'fish_per_person_hour' (fish_caught / (persons * hours))
    """
    df = df.copy()

    # Ensure numeric columns where appropriate; coerce invalid to NaN
    numeric_cols = ['persons', 'child', 'livebait', 'hours', 'camper', 'fish_caught']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            # If a required column is missing, create it as NaN so downstream will drop
            df[c] = np.nan

    # Drop rows missing crucial variables: outcome, exposure, and persons
    df = df.dropna(subset=['fish_caught', 'hours', 'persons'])

    # Remove implausible values
    df = df[df['hours'] > 0]
    df = df[df['persons'] > 0]

    # Binary indicators: treat NaN as 0 (assume missing -> no), then cast to int 0/1
    for bcol in ['livebait', 'child']:
        # If values beyond {0,1} exist, keep them but cast to int (preserve original intent)
        if bcol in df.columns:
            df[bcol] = df[bcol].fillna(0).astype(int)

    # Ensure camper is integer (if present)
    if 'camper' in df.columns:
        df['camper'] = df['camper'].fillna(0).astype(int)

    # Derived metrics
    df['fish_per_hour'] = df['fish_caught'] / df['hours']
    df['fish_per_person_hour'] = df['fish_caught'] / (df['persons'] * df['hours'])

    # Replace infinite values with NaN and drop rows where derived metrics are invalid
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['fish_per_hour', 'fish_per_person_hour'])

    # Log of hours for offset
    df['log_hours'] = np.log(df['hours'])

    # Ensure final dataframe contains the required columns (will raise KeyError upstream if not)
    required_cols = [
        'livebait', 'camper', 'persons', 'child',
        'fish_caught', 'hours', 'log_hours',
        'fish_per_hour', 'fish_per_person_hour'
    ]
    # Reorder/preserve only the required columns plus any others (do not remove required ones)
    # Keep all columns but ensure required ones exist
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Transformed dataframe is missing required columns: {missing}")

    return df


def model(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Fit a Negative Binomial GLM modeling total fish_caught with log(hours) as an offset.
    Predictors: livebait, camper, persons, child.

    Expects df to be the output of transform(...) and to contain the exact columns described
    in the transform docstring.
    Returns a dictionary with model object, textual summary, IRRs, and IRR confidence intervals.
    """
    # Validate required columns
    required_cols = ['fish_caught', 'hours', 'log_hours', 'livebait', 'camper', 'persons', 'child']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Model input dataframe is missing required columns: {missing}")

    # Prepare design matrix
    predictors = ['livebait', 'camper', 'persons', 'child']
    X = df[predictors].astype(float)
    X = sm.add_constant(X, has_constant='add')

    y = df['fish_caught'].astype(float)
    offset = df['log_hours'].astype(float)

    # Fit Negative Binomial GLM with offset
    model_fit = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset).fit()

    # Compute incident rate ratios (IRRs) and 95% CI
    params = model_fit.params
    conf = model_fit.conf_int()
    irrs = np.exp(params)
    # conf is a DataFrame with two columns (lower, upper)
    irrs_ci = np.exp(conf)

    results = {
        'model': model_fit,
        'summary': model_fit.summary().as_text(),
        'irrs': irrs.to_dict(),
        'irrs_ci': {k: (float(irrs_ci.loc[k, 0]), float(irrs_ci.loc[k, 1])) for k in irrs_ci.index}
    }

    return results