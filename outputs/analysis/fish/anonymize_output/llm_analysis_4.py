from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/fish/anonymize_output/fish.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Rename columns to meaningful names used in the model, if those original names exist
    df = df.rename(columns={
        'feature1': 'fish_count',
        'feature2': 'livebait',
        'feature3': 'camper',
        'feature4': 'n_adults',
        'feature5': 'n_children',
        'feature6': 'hours'
    }).copy()

    # Required final columns
    required = ['fish_count', 'livebait', 'camper', 'n_adults', 'n_children', 'hours']

    # Ensure required columns exist in the dataframe (create as NaN if missing)
    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    # Ensure numeric types where appropriate (coerce non-convertible to NaN)
    for col in required:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing critical values (now safe because required columns all exist)
    df = df.dropna(subset=required)

    # Remove rows with non-positive hours (cannot use as exposure)
    df = df[df['hours'] > 0].copy()

    # Ensure counts are non-negative integers
    df = df[df['fish_count'] >= 0].copy()
    # Round counts to nearest integer if numeric but not integer (safety)
    df['fish_count'] = df['fish_count'].round().astype(int)

    # Cast binary indicators to integers 0/1
    df['livebait'] = df['livebait'].round().astype(int)
    df['camper'] = df['camper'].round().astype(int)

    # Cast group counts to integers
    df['n_adults'] = df['n_adults'].round().astype(int)
    df['n_children'] = df['n_children'].round().astype(int)

    # Create derived variables useful for description or alternative models
    df['total_people'] = df['n_adults'] + df['n_children']
    # Observed rate per hour for descriptive statistics
    df['rate_per_hour'] = df['fish_count'] / df['hours']

    # Final columns required for modeling are: fish_count, livebait, camper, n_adults, n_children, hours
    # Return df with these columns plus helpful derived vars
    keep_cols = ['fish_count', 'livebait', 'camper', 'n_adults', 'n_children', 'hours', 'total_people', 'rate_per_hour']
    # Some derived columns may not exist if dataset lacked inputs, so filter
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a count model for fish_count using hours as exposure (offset).
    Steps:
    1. Fit Poisson GLM with log(hours) as offset.
    2. Compute dispersion (Pearson chi-square / df_resid). If overdispersion is detected (dispersion > 1.5), fit a Negative Binomial GLM as an alternative.
    3. Return fitted results and diagnostics.
    """
    # Required predictors
    predictors = ['livebait', 'camper', 'n_adults', 'n_children']
    for p in predictors:
        if p not in df.columns:
            raise ValueError(f"Required predictor column '{p}' not found in dataframe")
    if 'fish_count' not in df.columns or 'hours' not in df.columns:
        raise ValueError("Required columns 'fish_count' and 'hours' must be present in dataframe")

    # If no observations after transform, return a safe result without attempting to fit
    if df.shape[0] == 0:
        return {
            'poisson_result': None,
            'dispersion': np.nan,
            'negative_binomial_result': None,
            'notes': 'No observations available after transform; no model was fitted.',
            'summary': {
                'n_obs': 0,
                'poisson_aic': None,
                'poisson_deviance': None
            }
        }

    # Prepare endog and exog
    endog = df['fish_count'].astype(int)
    exog = df[predictors].astype(float)
    exog = sm.add_constant(exog, has_constant='add')

    # Offset = log(hours)
    offset = np.log(df['hours'].astype(float))

    results = {
        'poisson_result': None,
        'dispersion': np.nan,
        'negative_binomial_result': None,
        'notes': ''
    }

    # Fit Poisson GLM with offset
    try:
        poisson_model = sm.GLM(endog, exog, family=sm.families.Poisson(), offset=offset)
        poisson_res = poisson_model.fit()
        results['poisson_result'] = poisson_res
    except Exception as e:
        results['notes'] = f'Poisson model fit failed: {e}'
        results['summary'] = {
            'n_obs': int(len(endog)),
            'poisson_aic': None,
            'poisson_deviance': None
        }
        return results

    # Compute dispersion: Pearson chi2 / df_resid
    try:
        pearson_chi2 = float(np.sum(poisson_res.resid_pearson**2))
        df_resid = float(poisson_res.df_resid) if hasattr(poisson_res, 'df_resid') else np.nan
        if df_resid > 0:
            dispersion = pearson_chi2 / df_resid
        else:
            dispersion = np.nan
    except Exception:
        dispersion = np.nan

    results['dispersion'] = float(dispersion) if not np.isnan(dispersion) else np.nan

    # If substantial overdispersion, fit Negative Binomial as alternative
    # Threshold 1.5 is a heuristic; adjust as desired
    if not np.isnan(dispersion) and dispersion > 1.5:
        try:
            nb_model = sm.GLM(endog, exog, family=sm.families.NegativeBinomial(), offset=offset)
            nb_res = nb_model.fit()
            results['negative_binomial_result'] = nb_res
            results['notes'] = 'Overdispersion detected (dispersion > 1.5); negative binomial fitted.'
        except Exception as e:
            results['notes'] = f'Overdispersion detected but negative binomial fit failed: {e}'
    else:
        results['notes'] = 'Poisson model used (no strong overdispersion detected).' if not np.isnan(dispersion) else 'Dispersion not available; only Poisson fit returned.'

    # Add simple descriptive summaries to results for convenience
    summary = {
        'n_obs': int(poisson_res.nobs) if hasattr(poisson_res, 'nobs') else int(len(endog)),
        'poisson_aic': float(poisson_res.aic) if hasattr(poisson_res, 'aic') else None,
        'poisson_deviance': float(poisson_res.deviance) if hasattr(poisson_res, 'deviance') else None
    }
    if results['negative_binomial_result'] is not None:
        nbres = results['negative_binomial_result']
        summary['negbin_aic'] = float(nbres.aic) if hasattr(nbres, 'aic') else None
        summary['negbin_deviance'] = float(nbres.deviance) if hasattr(nbres, 'deviance') else None

    results['summary'] = summary

    return results