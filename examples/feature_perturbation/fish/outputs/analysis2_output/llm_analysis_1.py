from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the modeling dataframe. The function:
    - Renames columns to meaningful names (if present in the raw data)
    - Ensures required final columns exist
    - Handles missing or invalid essential values (FishCaught, Hours) with sensible defaults
    - Ensures binary variables are integers (0/1)
    - Creates GroupSize and PropChildren derived columns
    - Creates FishPerHour as a descriptive column

    Returns a dataframe with at least these columns:
    ['FishCaught', 'UsedLiveBait', 'HadCamper', 'NumAdults', 'NumChildren', 'GroupSize', 'PropChildren', 'Hours', 'FishPerHour']
    """
    df = df.copy()

    # Map possible raw feature names to required final names if present.
    rename_map = {
        'feature1': 'FishCaught',
        'feature2': 'UsedLiveBait',
        'feature3': 'HadCamper',
        'feature4': 'NumAdults',
        'feature5': 'NumChildren',
        'feature6': 'Hours'
    }
    # Only rename columns that exist in the dataframe to avoid creating unexpected NaNs
    existing_renames = {k: v for k, v in rename_map.items() if k in df.columns}
    if existing_renames:
        df = df.rename(columns=existing_renames)

    # Ensure all required final columns exist in the dataframe (create as NA if missing)
    final_cols = [
        'FishCaught',
        'UsedLiveBait',
        'HadCamper',
        'NumAdults',
        'NumChildren',
        'GroupSize',
        'PropChildren',
        'Hours',
        'FishPerHour'
    ]
    for col in ['FishCaught', 'UsedLiveBait', 'HadCamper', 'NumAdults', 'NumChildren', 'Hours']:
        if col not in df.columns:
            df[col] = pd.NA

    # Convert numeric fields to numeric types where appropriate
    df['FishCaught'] = pd.to_numeric(df['FishCaught'], errors='coerce')
    df['Hours'] = pd.to_numeric(df['Hours'], errors='coerce')
    # Binary indicators might be encoded as strings or numbers
    df['UsedLiveBait'] = pd.to_numeric(df['UsedLiveBait'], errors='coerce')
    df['HadCamper'] = pd.to_numeric(df['HadCamper'], errors='coerce')
    # Counts of people
    df['NumAdults'] = pd.to_numeric(df['NumAdults'], errors='coerce')
    df['NumChildren'] = pd.to_numeric(df['NumChildren'], errors='coerce')

    # Handle missing essential outcome/exposure sensibly:
    # - If FishCaught is missing, assume 0 (no fish recorded)
    # - If Hours is missing or non-positive, set to a default positive value (1.0 hour)
    df['FishCaught'] = df['FishCaught'].fillna(0)

    # Default hours for missing or non-positive values
    default_hours = 1.0
    # Replace NaN with default
    df['Hours'] = df['Hours'].fillna(default_hours)
    # Replace non-positive hours with default
    # Convert to float safely for comparison
    df['Hours'] = df['Hours'].astype(float)
    df.loc[df['Hours'] <= 0, 'Hours'] = default_hours

    # For remaining missing binary or count fields, fill sensible defaults:
    # - UsedLiveBait, HadCamper: default to 0 (assume not used / not present if unknown)
    # - NumAdults, NumChildren: default to 0
    df['UsedLiveBait'] = df['UsedLiveBait'].fillna(0)
    df['HadCamper'] = df['HadCamper'].fillna(0)
    df['NumAdults'] = df['NumAdults'].fillna(0)
    df['NumChildren'] = df['NumChildren'].fillna(0)

    # Ensure binary indicators are 0/1 integers
    def to_binary(x):
        try:
            return 1 if float(x) != 0 else 0
        except Exception:
            return 0

    df['UsedLiveBait'] = df['UsedLiveBait'].apply(to_binary).astype(int)
    df['HadCamper'] = df['HadCamper'].apply(to_binary).astype(int)

    # Ensure number of people are non-negative integers
    def to_nonneg_int(x):
        try:
            val = float(x)
            if np.isnan(val) or val <= 0:
                return 0
            return int(round(val))
        except Exception:
            return 0

    df['NumAdults'] = df['NumAdults'].apply(to_nonneg_int).astype(int)
    df['NumChildren'] = df['NumChildren'].apply(to_nonneg_int).astype(int)

    # Derive group size and proportion of children
    df['GroupSize'] = df['NumAdults'] + df['NumChildren']
    # Avoid division by zero for groups of size 0 -> set PropChildren = 0 when GroupSize == 0
    df['PropChildren'] = df.apply(lambda r: (r['NumChildren'] / r['GroupSize']) if r['GroupSize'] > 0 else 0.0, axis=1)

    # Derive fish per hour for descriptive purposes
    # Ensure FishCaught numeric
    df['FishCaught'] = df['FishCaught'].astype(float)
    # Hours already numeric and positive after previous steps
    df['Hours'] = df['Hours'].astype(float)
    # Avoid division by zero since Hours > 0 after imputation
    df['FishPerHour'] = df['FishCaught'] / df['Hours']

    # Reorder / ensure final columns exist
    for c in final_cols:
        if c not in df.columns:
            df[c] = pd.NA

    return df[final_cols]


def model(df: pd.DataFrame) -> Any:
    """
    Fit count regression models to estimate rate of fish caught per hour and the effect
    of predictors. The function performs the following:
    - Expects df to be the transformed dataframe returned by transform()
    - Fits a Poisson GLM with log(Hours) as an offset
    - Computes a dispersion statistic to check for overdispersion
    - If overdispersion is present (dispersion > 1.5) also fits a Negative Binomial GLM
    - Returns a dictionary with fitted results and diagnostics

    Returned dictionary keys:
    - 'poisson': fitted Poisson GLMResults
    - 'negbin': fitted Negative Binomial GLMResults (may be None if not fitted)
    - 'dispersion': dispersion statistic (deviance / df_resid) from Poisson
    - 'chosen': string 'negbin' if Negative Binomial used due to overdispersion else 'poisson'
    """
    # Work on a copy
    df = df.copy()

    # Ensure required columns exist
    required = ['FishCaught', 'UsedLiveBait', 'HadCamper', 'GroupSize', 'PropChildren', 'Hours']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns in dataframe for modeling: {missing}")

    # Drop rows with missing values in modeling columns (transform should have handled most cases)
    df = df.dropna(subset=required)

    # Small sanity checks
    # Ensure Hours positive
    df = df[df['Hours'].astype(float) > 0]

    # If no data remain after cleaning, raise an informative error
    if df.shape[0] == 0:
        raise ValueError("No data available for modeling after preprocessing (no rows).")

    # Build formula: model fish counts with exposure (Hours) as offset
    formula = 'FishCaught ~ UsedLiveBait + HadCamper + GroupSize + PropChildren'

    # Offset is log(hours) to model rate per hour
    offset = np.log(df['Hours'].astype(float))
    # Make offset a pandas Series aligned with df.index to avoid alignment issues
    offset = pd.Series(offset.values, index=df.index)

    # Fit Poisson GLM. Use a robust fitting strategy with fallbacks if the primary approach fails.
    poisson_results = None
    fit_error = None

    # Primary attempt: formula-based interface
    try:
        poisson_model = smf.glm(formula=formula, data=df, family=sm.families.Poisson(), offset=offset)
        poisson_results = poisson_model.fit()
    except Exception as e_primary:
        fit_error = e_primary
        # Fallback 1: build design matrix manually and fit via sm.GLM with explicit start_params
        try:
            # Prepare design matrix (including intercept)
            X = df[['UsedLiveBait', 'HadCamper', 'GroupSize', 'PropChildren']].astype(float)
            X = sm.add_constant(X, has_constant='add')
            y = df['FishCaught'].astype(float)

            # Replace any infinite values in X or offset
            if not np.isfinite(X.values).all():
                X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if not np.isfinite(offset.values).all():
                offset = pd.Series(np.where(np.isfinite(offset.values), offset.values, 0.0), index=offset.index)

            # Try fitting with zeros as starting params
            glm_model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
            start_params = np.zeros(X.shape[1])
            poisson_results = glm_model.fit(start_params=start_params, maxiter=100)
        except Exception as e_fallback1:
            fit_error = e_fallback1
            # Fallback 2: if counts are all zero, fitting Poisson by maximum likelihood may be problematic.
            # In that case, we can fit a model on tiny jittered counts to allow convergence, then report.
            try:
                y = df['FishCaught'].astype(float)
                if y.sum() == 0:
                    # Add a tiny constant jitter so that deviance / initial guesses are finite.
                    y_jitter = y + 1e-8
                    X = df[['UsedLiveBait', 'HadCamper', 'GroupSize', 'PropChildren']].astype(float)
                    X = sm.add_constant(X, has_constant='add')
                    glm_model = sm.GLM(y_jitter, X, family=sm.families.Poisson(), offset=offset)
                    poisson_results = glm_model.fit(start_params=np.zeros(X.shape[1]), maxiter=100)
                else:
                    # General jitter to break pathological ties
                    y_jitter = df['FishCaught'].astype(float) + np.random.uniform(0, 1e-8, size=df.shape[0])
                    X = df[['UsedLiveBait', 'HadCamper', 'GroupSize', 'PropChildren']].astype(float)
                    X = sm.add_constant(X, has_constant='add')
                    glm_model = sm.GLM(y_jitter, X, family=sm.families.Poisson(), offset=offset)
                    poisson_results = glm_model.fit(start_params=np.zeros(X.shape[1]), maxiter=100)
            except Exception as e_fallback2:
                fit_error = e_fallback2
                poisson_results = None

    if poisson_results is None:
        # If after all attempts we failed, surface an informative error.
        raise RuntimeError(f"Poisson GLM failed to fit after multiple attempts: {fit_error}")

    # Compute dispersion statistic: deviance / df_resid
    disp = (
        float(poisson_results.deviance) / float(poisson_results.df_resid)
        if getattr(poisson_results, "df_resid", None) is not None and poisson_results.df_resid > 0
        else np.nan
    )

    # Default: do not fit Negative Binomial unless overdispersion detected
    negbin_results = None
    chosen = 'poisson'

    if np.isfinite(disp) and disp > 1.5:
        try:
            # Attempt Negative Binomial via formula interface first
            nb_model = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial(), offset=offset)
            negbin_results = nb_model.fit()
            chosen = 'negbin'
        except Exception:
            # Fallback: GLM with design matrix
            try:
                X = df[['UsedLiveBait', 'HadCamper', 'GroupSize', 'PropChildren']].astype(float)
                X = sm.add_constant(X, has_constant='add')
                y = df['FishCaught'].astype(float)
                glm_nb = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset)
                negbin_results = glm_nb.fit()
                chosen = 'negbin'
            except Exception:
                negbin_results = None
                chosen = 'poisson'

    return {
        'poisson': poisson_results,
        'negbin': negbin_results,
        'dispersion': float(disp) if np.isfinite(disp) else None,
        'chosen': chosen
    }