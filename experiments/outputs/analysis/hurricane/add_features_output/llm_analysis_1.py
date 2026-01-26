from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load dataset (path kept from original code)
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/add_features_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and transform the hurricane dataset for analysis.

    Outputs (columns required by the statistical model):
      - alldeaths: integer count of fatalities (kept as-is for count model)
      - log_alldeaths: log(alldeaths + 1) for OLS robustness check
      - masfem_z: standardized masculinity-femininity index (z-scored from 'masfem')
      - wind, category, min, year, elapsedyrs: numeric controls

    The function coerces key columns to numeric, drops rows with missing values
    in the variables required for the main models, and creates derived columns.
    """
    df = df.copy()

    # Ensure relevant columns exist and coerce to numeric where appropriate
    numeric_cols = ['alldeaths', 'masfem', 'wind', 'category', 'min', 'year', 'elapsedyrs', 'masfem_mturk', 'ndam15']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing the dependent variable or the main independent variable or core controls
    required = ['alldeaths', 'masfem', 'wind', 'category', 'min', 'year', 'elapsedyrs']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input dataframe: {missing}")

    df = df.dropna(subset=required).reset_index(drop=True)

    # Ensure alldeaths is an integer count
    # If alldeaths is float due to upstream formatting, cast to int after rounding (but preserve zeros)
    df['alldeaths'] = df['alldeaths'].round().astype(int)

    # Create log-transformed deaths for OLS robustness
    df['log_alldeaths'] = np.log(df['alldeaths'] + 1)

    # Create a binary indicator for any deaths (useful if doing alternative logistic models)
    df['deaths_binary'] = (df['alldeaths'] > 0).astype(int)

    # Standardize the masfem index for interpretability and numerical stability
    # Use population std (ddof=0) to avoid potential division by zero
    masfem_std = df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1.0
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / masfem_std

    # Return full dataframe copy with new columns (ensures required final columns are present)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit the main statistical models testing whether more feminine hurricane names (higher masfem)
    are associated with fewer precautionary measures as proxied by fatalities.

    Primary model: Negative binomial regression on raw death counts (alldeaths) to account for
    count data and over-dispersion.

    Robustness check: OLS on log(alldeaths + 1) with robust standard errors.

    Returns a dictionary with fitted model objects and robust-covariance-adjusted results.
    """
    # Work on a copy
    df = df.copy()

    # Verify required columns exist (helps surface clear errors)
    required_final_cols = ['alldeaths', 'log_alldeaths', 'masfem_z', 'wind', 'category', 'min', 'year', 'elapsedyrs']
    missing_final = [c for c in required_final_cols if c not in df.columns]
    if missing_final:
        raise ValueError(f"Transformed dataframe is missing required columns for modeling: {missing_final}")

    # Define formula for main covariates. We use the standardized masfem index 'masfem_z'.
    formula = 'alldeaths ~ masfem_z + wind + category + min + year + elapsedyrs'

    # Negative binomial (accounts for over-dispersed counts). Using GLM with NegativeBinomial family.
    # Fit base model
    nb_model = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial()).fit()
    # Fit again requesting robust covariance (some statsmodels GLM results may not support get_robustcov_results)
    # so we obtain a results object with robust covariances by specifying cov_type at fit time.
    nb_model_robust = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial()).fit(cov_type='HC3')

    # Robustness check: OLS on log transformed deaths
    ols_formula = 'log_alldeaths ~ masfem_z + wind + category + min + year + elapsedyrs'
    ols_model = smf.ols(formula=ols_formula, data=df).fit()
    # For OLS, use get_robustcov_results which is available on RegressionResults
    ols_model_robust = ols_model.get_robustcov_results(cov_type='HC3')

    # Pack results. Callers can inspect .summary() or use the returned fitted-objects directly.
    results = {
        'nb_model': nb_model,
        'nb_model_robust': nb_model_robust,
        'ols_log_model': ols_model,
        'ols_log_model_robust': ols_model_robust
    }

    return results