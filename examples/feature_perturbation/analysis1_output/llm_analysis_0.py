from typing import Any, Dict
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataframe into the analytic dataset used for modeling.

    Steps:
    - Ensure relevant columns are numeric / categorical as needed.
    - Drop rows missing the dependent variable or the primary IV or essential controls.
    - Standardize the masfem measure to masfem_z (mean 0, sd 1).
    - Create a mean-centered year variable year_centered.
    - Create a log-deaths (+1) column for OLS robustness checks.

    The final returned dataframe contains the exact columns referenced in the statistical model.
    """
    # Make a copy to avoid modifying in place
    df = df.copy()

    # Ensure key numeric columns are numeric
    numeric_cols = [
        'masfem',
        'alldeaths',
        'wind',
        'min',
        'category',
        'year',
        'elapsedyrs',
        'masfem_mturk',
        'ndam15',
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Ensure source is categorical if present
    if 'source' in df.columns:
        df['source'] = df['source'].astype('category')

    # Drop rows missing dependent variable or primary IV or core controls
    required_columns = ['alldeaths', 'masfem', 'wind', 'min', 'category', 'year']
    df = df.dropna(subset=[c for c in required_columns if c in df.columns])

    # Standardize masfem -> masfem_z used in model
    if 'masfem' in df.columns:
        mas_mean = df['masfem'].mean()
        mas_std = df['masfem'].std(ddof=0)
        if pd.isna(mas_std) or mas_std == 0:
            mas_std = 1.0
        df['masfem_z'] = (df['masfem'] - mas_mean) / mas_std
    else:
        # If masfem missing entirely, create column of nans to keep contract intact
        df['masfem_z'] = np.nan

    # Center year to aid interpretation and numerical stability
    if 'year' in df.columns:
        year_mean = df['year'].mean()
        df['year_centered'] = df['year'] - year_mean
    else:
        df['year_centered'] = np.nan

    # Create log(alldeaths + 1) for OLS robustness checks
    if 'alldeaths' in df.columns:
        df['log_alldeaths_plus1'] = np.log1p(df['alldeaths'].fillna(0))
    else:
        df['log_alldeaths_plus1'] = np.nan

    # Ensure category is treated as a categorical variable for modeling (Saffir-Simpson)
    if 'category' in df.columns:
        # convert to numeric first (coerce invalid to NaN), then to pandas categorical
        df['category'] = pd.to_numeric(df['category'], errors='coerce')
        df['category'] = df['category'].astype('category')

    # Keep only columns needed for modeling + helpful diagnostics
    keep_cols = []
    for c in [
        'alldeaths',
        'masfem_z',
        'wind',
        'min',
        'category',
        'year_centered',
        'elapsedyrs',
        'source',
        'log_alldeaths_plus1',
        'masfem',
        'masfem_mturk',
    ]:
        if c in df.columns:
            keep_cols.append(c)

    df = df[keep_cols].reset_index(drop=True)

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit statistical models to estimate the association between perceived femininity of hurricane names
    and fatalities, controlling for storm intensity and time.

    Primary model: Negative binomial GLM on raw death counts (alldeaths) because the outcome is a count and highly
    overdispersed (many zeros, heavy right tail).

    Robustness model: OLS on log(alldeaths + 1) with robust standard errors.

    Returns:
    - a dict containing the fitted Negative Binomial result ('nb_result') and the OLS result ('ols_result') when available.
    """
    results: Dict[str, Any] = {}

    # Ensure required columns exist
    required = ['alldeaths', 'masfem_z', 'wind', 'min', 'category', 'year_centered']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Drop any remaining rows with NA in model variables
    model_df = df.dropna(subset=['alldeaths', 'masfem_z', 'wind', 'min', 'category', 'year_centered'])

    # Primary specification: Negative Binomial regression for counts
    formula_terms = ['masfem_z', 'wind', 'min', 'C(category)', 'year_centered']
    if 'elapsedyrs' in model_df.columns:
        formula_terms.append('elapsedyrs')
    if 'source' in model_df.columns:
        formula_terms.append('C(source)')

    formula = 'alldeaths ~ ' + ' + '.join(formula_terms)

    # Fit Negative Binomial GLM
    nb_model = smf.glm(formula=formula, data=model_df, family=sm.families.NegativeBinomial())
    nb_result_raw = nb_model.fit()

    # Attempt to obtain robust covariance results; fall back to raw results if not available
    if hasattr(nb_result_raw, 'get_robustcov_results'):
        try:
            nb_result = nb_result_raw.get_robustcov_results(cov_type='HC3')
        except Exception:
            warnings.warn("Could not compute robust covariance for NB model; returning raw NB results.", RuntimeWarning)
            nb_result = nb_result_raw
    else:
        # Some versions of statsmodels may not expose get_robustcov_results on GLMResults;
        # in that case, return the raw fit object.
        warnings.warn("GLMResults has no method get_robustcov_results; returning raw NB results.", RuntimeWarning)
        nb_result = nb_result_raw

    results['nb_result'] = nb_result

    # Robustness: OLS on log(alldeaths + 1)
    if 'log_alldeaths_plus1' in model_df.columns:
        ols_formula = 'log_alldeaths_plus1 ~ ' + ' + '.join(formula_terms)
        ols_model = smf.ols(formula=ols_formula, data=model_df)
        ols_result_raw = ols_model.fit()
        if hasattr(ols_result_raw, 'get_robustcov_results'):
            try:
                ols_result = ols_result_raw.get_robustcov_results(cov_type='HC3')
            except Exception:
                warnings.warn("Could not compute robust covariance for OLS model; returning raw OLS results.", RuntimeWarning)
                ols_result = ols_result_raw
        else:
            warnings.warn("OLS results object has no method get_robustcov_results; returning raw OLS results.", RuntimeWarning)
            ols_result = ols_result_raw

        results['ols_result'] = ols_result

    return results