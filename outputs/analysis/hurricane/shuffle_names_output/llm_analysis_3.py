from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a dataframe with the variables used in the models.

    Produces the following columns used in modeling:
      - Deaths: raw count of deaths (from 'ndam15')
      - LogDeaths: log(Deaths + 1)
      - NameFemininity: continuous masculinity-femininity index (from 'name')
      - NameFem_z: standardized NameFemininity (mean 0, sd 1) for descriptives/robustness
      - FemaleName: binary indicator from 'elapsedyrs' (0 male name, 1 female name)
      - MaxWind: maximum wind speed (from 'wind')
      - Category: Saffir–Simpson category (from 'masfem')
      - Year: year of storm (from 'alldeaths')
      - LogDamage: log(ind + 1) (economic damage control)

    The function coerces types and fills reasonable defaults so that the
    modeling function receives the required final columns. It avoids
    dropping all rows aggressively; instead it imputes missing control
    values conservatively (median or 0) and treats missing deaths as 0.
    """
    df = df.copy()

    # Ensure input columns exist (raise if entirely missing)
    required_inputs = {
        'ndam15': "Expected column 'ndam15' with death counts not found in dataframe",
        'name': "Expected column 'name' (masculinity-femininity index) not found in dataframe",
        'elapsedyrs': "Expected column 'elapsedyrs' (binary female-name indicator) not found in dataframe",
        'wind': "Expected column 'wind' (max wind speed) not found in dataframe",
        'masfem': "Expected column 'masfem' (category) not found in dataframe",
        'alldeaths': "Expected column 'alldeaths' (year) not found in dataframe",
        'ind': "Expected column 'ind' (normalized damage) not found in dataframe"
    }
    for col, err in required_inputs.items():
        if col not in df.columns:
            raise KeyError(err)

    # Coerce to numeric where appropriate
    df['Deaths'] = pd.to_numeric(df['ndam15'], errors='coerce')
    # Treat missing death counts as 0 and negative as 0 (can't have negative fatalities)
    df.loc[df['Deaths'].isna(), 'Deaths'] = 0
    df.loc[df['Deaths'] < 0, 'Deaths'] = 0
    # Ensure integer counts
    df['Deaths'] = df['Deaths'].astype(int)

    # Name femininity - continuous. Coerce and if entirely missing, set to 0.
    df['NameFemininity'] = pd.to_numeric(df['name'], errors='coerce')
    if df['NameFemininity'].isna().all():
        df['NameFemininity'] = 0.0
    else:
        mean_name = df['NameFemininity'].mean(skipna=True)
        df['NameFemininity'] = df['NameFemininity'].fillna(mean_name)

    # FemaleName binary indicator - coerce then binarize (non-zero -> 1)
    df['FemaleName'] = pd.to_numeric(df['elapsedyrs'], errors='coerce').fillna(0)
    df['FemaleName'] = (df['FemaleName'] != 0).astype(int)

    # MaxWind - coerce, fill missing with median if available, else 0
    df['MaxWind'] = pd.to_numeric(df['wind'], errors='coerce')
    if df['MaxWind'].notna().any():
        df['MaxWind'] = df['MaxWind'].fillna(df['MaxWind'].median())
    else:
        df['MaxWind'] = df['MaxWind'].fillna(0.0)

    # Category - coerce, fill missing with median if available, else 0
    df['Category'] = pd.to_numeric(df['masfem'], errors='coerce')
    if df['Category'].notna().any():
        median_cat = df['Category'].median()
        df['Category'] = df['Category'].fillna(median_cat)
    else:
        df['Category'] = df['Category'].fillna(0)

    # Year - coerce, fill missing with median year if available, else 0
    df['Year'] = pd.to_numeric(df['alldeaths'], errors='coerce')
    if df['Year'].notna().any():
        median_year = df['Year'].median()
        # Round median year to nearest int for imputation
        df['Year'] = df['Year'].fillna(int(round(median_year)))
    else:
        df['Year'] = df['Year'].fillna(0)
    # Ensure Year is integer
    df['Year'] = df['Year'].astype(int)

    # LogDamage: coerce normalized damage, negative -> NaN, then log1p with missing -> 0
    df['LogDamage'] = pd.to_numeric(df['ind'], errors='coerce')
    df.loc[df['LogDamage'] < 0, 'LogDamage'] = np.nan
    df['LogDamage'] = np.log(df['LogDamage'].fillna(0) + 1)

    # Log-transform of deaths for OLS robustness
    df['LogDeaths'] = np.log(df['Deaths'] + 1)

    # Standardize name femininity for descriptives/interpretation
    name_mean = df['NameFemininity'].mean()
    name_std = df['NameFemininity'].std(ddof=0)
    if pd.isna(name_std) or name_std == 0:
        df['NameFem_z'] = 0.0
    else:
        df['NameFem_z'] = (df['NameFemininity'] - name_mean) / name_std

    # Final required columns (must match the conceptual variables)
    required_cols = ['Deaths', 'LogDeaths', 'NameFemininity', 'NameFem_z', 'FemaleName', 'MaxWind', 'Category', 'Year', 'LogDamage']

    # Ensure these columns are present; they should be after the transformations above
    missing_final = [c for c in required_cols if c not in df.columns]
    if missing_final:
        raise RuntimeError(f"After transform, required columns are missing: {missing_final}")

    # Return only the required columns in the specified order
    return df[required_cols].reset_index(drop=True)


def model(df: pd.DataFrame) -> Any:
    """
    Fit two models assessing the relationship between feminine hurricane names and fatalities:
      1) Negative-binomial GLM on raw count of deaths (primary model for count outcome).
      2) OLS on log(Deaths + 1) as a robustness check.

    Independent variables: NameFemininity (continuous) and FemaleName (binary), controlling for MaxWind, Category, LogDamage, and Year.

    Returns a dictionary with statsmodels results objects: {'nb_result': nb_res, 'ols_result': ols_res}
    """
    needed = ['Deaths', 'LogDeaths', 'NameFemininity', 'FemaleName', 'MaxWind', 'Category', 'LogDamage', 'Year']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Input dataframe is missing required columns for modeling: {missing}")

    if df.shape[0] == 0:
        raise ValueError("Input dataframe contains no rows after transform; cannot fit models.")

    # Design matrix with the specified independent variables / controls
    X = df[['NameFemininity', 'FemaleName', 'MaxWind', 'Category', 'LogDamage', 'Year']].copy()

    # Detect and drop predictors with zero variance (constant columns).
    const_cols = []
    for col in X.columns:
        if X[col].nunique(dropna=False) <= 1:
            const_cols.append(col)
    if const_cols:
        print(f"Warning: dropping constant predictor(s) before model fitting: {const_cols}")
        X = X.drop(columns=const_cols)

    # Add intercept
    X = sm.add_constant(X, has_constant='add')

    # Dependent variables
    y_count = df['Deaths'].values
    y_log = df['LogDeaths'].values

    # Validate that X and y have matching, non-zero shapes and contain finite values
    if X.shape[0] == 0 or y_count.size == 0:
        raise ValueError("No observations available for modeling after preparing design matrix.")

    # Drop any rows with non-finite values
    finite_mask = np.isfinite(y_count) & np.isfinite(y_log) & np.all(np.isfinite(X.values), axis=1)
    if not np.any(finite_mask):
        raise ValueError("No finite observations available for modeling after filtering.")
    if finite_mask.sum() < X.shape[0]:
        X = X.loc[finite_mask].copy()
        y_count = y_count[finite_mask]
        y_log = y_log[finite_mask]

    # Final check
    if X.shape[0] == 0 or y_count.size == 0:
        raise ValueError("No observations available for modeling after dropping non-finite rows.")

    # Helper to fit a model and obtain a results object with robust (HC3) covariance where possible
    def fit_with_robust(model_obj):
        """
        Try to fit the model with robust covariance. Approaches attempted in order:
          1) model_obj.fit(cov_type='HC3') if supported by the fit method.
          2) model_obj.fit() followed by .get_robustcov_results(cov_type='HC3') if available.
          3) model_obj.fit() and manually compute HC3 sandwich covariance and attach it to the results.
        Returns the results object (possibly modified in-place) that provides summary(), params, bse, etc.
        """
        try:
            # Preferred: ask fit to return results already using robust cov
            res = model_obj.fit(cov_type='HC3')
            return res
        except TypeError:
            # fit doesn't accept cov_type kwarg on this statsmodels version
            pass
        except Exception:
            # Other fit errors should be propagated to outer handlers
            raise

        # Fallback: fit normally
        res = model_obj.fit()

        # If get_robustcov_results exists, use it
        if hasattr(res, 'get_robustcov_results'):
            try:
                return res.get_robustcov_results(cov_type='HC3')
            except Exception:
                # If that fails, continue to manual sandwich
                pass

        # Manual sandwich: compute HC3 covariance and attach it to the results object.
        try:
            from statsmodels.stats.sandwich_covariance import cov_hc3
            robust_cov = cov_hc3(res)
            # Attach cov_params method
            try:
                res.cov_params = lambda: robust_cov
            except Exception:
                # If attribute assignment fails, ignore
                pass
            # Update bse, tvalues, pvalues if params available
            try:
                params = res.params
                bse = np.sqrt(np.diag(robust_cov))
                # Avoid division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    tvals = params / bse
                res.bse = bse
                res.tvalues = tvals
                # Compute p-values using normal approximation if scipy is available
                try:
                    from scipy import stats as _sps
                    res.pvalues = 2 * (1 - _sps.norm.cdf(np.abs(tvals)))
                except Exception:
                    # If scipy not available, leave pvalues as-is or absent
                    pass
            except Exception:
                pass
            return res
        except Exception:
            # If sandwich computation fails, return the plain results
            return res

    # 1) Negative-binomial GLM for count data (attempt robust cov via helper)
    nb_res = None
    try:
        nb_model = sm.GLM(y_count, X, family=sm.families.NegativeBinomial())
        nb_res = fit_with_robust(nb_model)
    except Exception as e_nb:
        # If NB fails to converge or errors out (e.g., singular matrix), attempt Poisson with robust SEs as fallback
        print(f"NegativeBinomial fit failed with error: {e_nb}. Attempting Poisson fallback.")
        try:
            pois_model = sm.GLM(y_count, X, family=sm.families.Poisson())
            nb_res = fit_with_robust(pois_model)
        except Exception as e_pois:
            raise RuntimeError(f"Negative-Binomial and Poisson model fitting both failed: {e_pois}")

    # 2) OLS on log-transformed deaths as robustness check
    try:
        ols_model = sm.OLS(y_log, X)
        ols_res = fit_with_robust(ols_model)
    except Exception as e:
        raise RuntimeError(f"OLS model fitting failed: {e}")

    # Print summaries for immediate inspection (optional)
    try:
        print('\n=== Negative-Binomial / Poisson (fallback) Results ===')
        print(nb_res.summary())
    except Exception:
        print('NB/Poisson model summary unavailable')

    print('\n=== OLS (log-deaths) Results ===')
    try:
        print(ols_res.summary())
    except Exception:
        print('OLS model summary unavailable')

    return {
        'nb_result': nb_res,
        'ols_result': ols_res
    }