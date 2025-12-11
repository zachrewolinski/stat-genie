from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial as NBDiscrete


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and derive variables needed for the statistical models.

    Produces standardized femininity score (masfem_z), log-transformed outcomes/controls,
    centered year, and retains the source categorical variable for dummying in the model stage.

    Ensures that the FINAL dataframe contains the exact required conceptual columns:
      ['masfem_z', 'alldeaths', 'log_alldeaths', 'wind', 'category', 'min',
       'ndam15_log', 'year_c', 'elapsedyrs', 'source', 'masfem_mturk', 'gender_mf']
    If optional robustness columns (masfem_mturk, gender_mf) are missing in the input they will
    be created with NaN so that the final dataframe has a consistent set of columns.
    """
    df = df.copy()

    # Ensure necessary input columns exist for computing required outputs
    required_input_cols = [
        'alldeaths', 'masfem', 'wind', 'category', 'min', 'ndam15', 'year', 'elapsedyrs', 'source'
    ]
    missing = [c for c in required_input_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for transform: {missing}")

    # Drop rows with missing critical values (keep masfem_mturk and gender_mf as optional robustness vars)
    df = df.dropna(subset=['alldeaths', 'masfem', 'wind', 'category', 'min', 'ndam15', 'year', 'source'])

    # Ensure numeric conversions for inputs used in transforms
    df['alldeaths'] = pd.to_numeric(df['alldeaths'], errors='coerce')
    df['masfem'] = pd.to_numeric(df['masfem'], errors='coerce')
    df['wind'] = pd.to_numeric(df['wind'], errors='coerce')
    df['category'] = pd.to_numeric(df['category'], errors='coerce')
    df['min'] = pd.to_numeric(df['min'], errors='coerce')
    df['ndam15'] = pd.to_numeric(df['ndam15'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['elapsedyrs'] = pd.to_numeric(df['elapsedyrs'], errors='coerce')

    # Create log-transformed outcome for OLS robustness (log1p handles zeros)
    # Guard against invalid values by coercing to finite numbers first
    df['log_alldeaths'] = np.log1p(df['alldeaths'].astype(float))

    # Log-transform the damage (ndam15) because it is highly skewed
    df['ndam15_log'] = np.log1p(df['ndam15'].astype(float))

    # Standardize masfem (z-score)
    masfem_mean = df['masfem'].mean()
    masfem_std = df['masfem'].std(ddof=0)
    if masfem_std == 0 or np.isnan(masfem_std):
        df['masfem_z'] = 0.0
    else:
        df['masfem_z'] = (df['masfem'] - masfem_mean) / masfem_std

    # Center year
    df['year_c'] = df['year'].astype(float) - df['year'].astype(float).mean()

    # Ensure optional robustness columns exist in the final dataframe (create if missing)
    if 'masfem_mturk' not in df.columns:
        df['masfem_mturk'] = np.nan
    else:
        df['masfem_mturk'] = pd.to_numeric(df['masfem_mturk'], errors='coerce')

    if 'gender_mf' not in df.columns:
        df['gender_mf'] = np.nan
    else:
        df['gender_mf'] = pd.to_numeric(df['gender_mf'], errors='coerce')

    # Ensure source is string (categorical)
    df['source'] = df['source'].astype(str)

    # Trim DataFrame to required final columns + a few originals for reference if present
    # Must include the exact conceptual variable names as specified
    keep_cols = [
        'alldeaths', 'log_alldeaths', 'masfem', 'masfem_z', 'masfem_mturk', 'gender_mf',
        'wind', 'category', 'min', 'ndam15', 'ndam15_log', 'year', 'year_c', 'elapsedyrs', 'source', 'name'
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    # Final drop of any rows with newly created NA in required numeric covariates for modeling
    df = df.dropna(subset=['masfem_z', 'wind', 'category', 'min', 'ndam15_log', 'year_c', 'elapsedyrs', 'alldeaths'])

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit two complementary models to evaluate the relationship between name femininity and fatalities:
      1) Negative Binomial regression predicting raw count alldeaths
      2) OLS regression predicting log1p(alldeaths) as a robustness check

    Returns a dict with the fitted results objects: {'nb_model': nb_res_robust, 'ols_model': ols_res_robust}

    The model expects the transformed FINAL dataframe produced by transform().
    """
    data = df.copy()

    # Required covariates for the model (must match conceptual variable column names)
    covariates = ['masfem_z', 'wind', 'category', 'min', 'ndam15_log', 'year_c', 'elapsedyrs']
    missing_covs = [c for c in covariates if c not in data.columns]
    if missing_covs:
        raise ValueError(f"Missing covariates in transformed df: {missing_covs}")

    # Coerce covariates and outcome to numeric, detect and remove rows with NaN/inf before modeling
    cov_df = data[covariates].apply(pd.to_numeric, errors='coerce')
    y_count = pd.to_numeric(data['alldeaths'], errors='coerce')

    # Build mask for finite values across covariates and outcome
    # np.isfinite on values ensures no infs
    cov_finite = np.isfinite(cov_df.values).all(axis=1)
    y_finite = np.isfinite(y_count.values)
    cov_notnull = cov_df.notnull().all(axis=1)
    y_notnull = y_count.notnull()

    valid_mask = cov_finite & y_finite & cov_notnull & y_notnull

    if not valid_mask.any():
        raise RuntimeError("No valid rows available for modeling after removing NaN/inf in covariates or outcome.")

    # Subset and reset index to ensure aligned inputs
    data = data.loc[valid_mask].reset_index(drop=True)
    cov_df = cov_df.loc[valid_mask].reset_index(drop=True)
    y_count = y_count.loc[valid_mask].reset_index(drop=True)

    X = cov_df.astype(float).copy()

    # Add dummies for source (drop first to avoid multicollinearity) but keep the original 'source' column in df
    if 'source' in data.columns:
        source_dummies = pd.get_dummies(data['source'].astype(str), prefix='source', drop_first=True)
        if source_dummies.shape[1] > 0:
            # reset_index on both to ensure positional concat
            X = pd.concat([X.reset_index(drop=True), source_dummies.reset_index(drop=True)], axis=1)

    # Ensure all exogenous variables are numeric floats and contain no NaN/Inf
    X = X.apply(pd.to_numeric, errors='coerce').astype(float)
    # Drop any remaining rows with NaN/inf in X or y_count (defensive)
    finite_X_mask = np.isfinite(X.values).all(axis=1)
    finite_y_mask = np.isfinite(y_count.values)
    final_mask = finite_X_mask & finite_y_mask
    if not final_mask.all():
        X = X.loc[final_mask].reset_index(drop=True)
        y_count = y_count.loc[final_mask].reset_index(drop=True)
        data = data.loc[final_mask].reset_index(drop=True)

    # Add constant term
    X = sm.add_constant(X, has_constant='add')

    # Helper wrapper to present a unified interface with robust covariance/properties
    class _RobustResultWrapper:
        def __init__(self, original_res, cov_matrix: np.ndarray):
            self._res = original_res
            # Ensure cov_matrix is a numpy array
            self._cov = np.asarray(cov_matrix)
            # Try to get params from the original or robust results
            self.params = getattr(original_res, 'params', None)
            # compute robust standard errors if possible
            try:
                diag = np.diag(self._cov)
                self.bse = np.sqrt(np.where(diag >= 0, diag, np.nan))
            except Exception:
                self.bse = None

        def cov_params(self):
            return self._cov

        def get_robustcov_results(self, cov_type='HC3'):
            return self

        def __getattr__(self, item):
            # Delegate attribute access to the original results object where possible
            return getattr(self._res, item)

    # Helper to obtain a robust-wrapped result object from a fitted results instance
    def _make_robust(res):
        # Prefer to use result.get_robustcov_results when available (it may handle internals better)
        try:
            if hasattr(res, 'get_robustcov_results'):
                robust_res = res.get_robustcov_results(cov_type='HC3')
                cov = robust_res.cov_params()
                params = getattr(robust_res, 'params', getattr(res, 'params', None))
                wrapper = _RobustResultWrapper(robust_res if robust_res is not None else res, cov)
                wrapper.params = params
                return wrapper
        except Exception:
            pass

        # Manual robust covariance calculation
        try:
            from statsmodels.stats.sandwich_covariance import cov_hc3
            cov = cov_hc3(res)
            wrapper = _RobustResultWrapper(res, cov)
            wrapper.params = getattr(res, 'params', None)
            return wrapper
        except Exception as e:
            # As a last resort, return the original results object if it provides cov_params
            if hasattr(res, 'cov_params'):
                try:
                    cov = res.cov_params()
                    wrapper = _RobustResultWrapper(res, cov)
                    wrapper.params = getattr(res, 'params', None)
                    return wrapper
                except Exception:
                    pass
            raise RuntimeError(f"Failed to construct robust results wrapper: {e}")

    # Fit Negative Binomial via GLM (preferred)
    nb_res_robust = None
    try:
        nb_model = sm.GLM(y_count, X, family=sm.families.NegativeBinomial())
        nb_res = nb_model.fit()
        nb_res_robust = _make_robust(nb_res)
    except Exception:
        # Fallback to Discrete negative binomial if GLM fails
        try:
            # NBDiscrete expects exog and endog; ensure shapes align by resetting indices
            X_nb = X.reset_index(drop=True)
            y_nb = y_count.reset_index(drop=True)
            nb_disc = NBDiscrete(y_nb, X_nb)
            nb_res = nb_disc.fit(disp=False)
            nb_res_robust = _make_robust(nb_res)
        except Exception as e:
            raise RuntimeError(f"Negative binomial model failed: {e}")

    # OLS robustness on log-transformed deaths
    if 'log_alldeaths' not in data.columns:
        data['log_alldeaths'] = np.log1p(data['alldeaths'].astype(float))
    y_log = pd.to_numeric(data['log_alldeaths'], errors='coerce').reset_index(drop=True)
    X_ols = X.reset_index(drop=True)
    # Ensure no NaN/inf in OLS inputs
    finite_mask_ols = np.isfinite(X_ols.values).all(axis=1) & np.isfinite(y_log.values)
    if not finite_mask_ols.all():
        X_ols = X_ols.loc[finite_mask_ols].reset_index(drop=True)
        y_log = y_log.loc[finite_mask_ols].reset_index(drop=True)
    ols_model = sm.OLS(y_log, X_ols)
    ols_res = ols_model.fit()
    ols_res_robust = _make_robust(ols_res)

    return {
        'nb_model': nb_res_robust,
        'ols_model': ols_res_robust
    }