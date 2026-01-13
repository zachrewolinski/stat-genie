from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare dataframe for modeling.

    Produces the following final columns used in the model:
      - ndam15: numeric total deaths (endogenous/count outcome)
      - name: original masculinity-femininity index (numeric)
      - name_c: mean-centered name index (primary IV)
      - elapsedyrs: original binary gender indicator (0 male, 1 female)
      - is_female_name: binary indicator derived from elapsedyrs
      - wind, min, ind: numeric controls for storm severity/damage
      - StormYear: numeric storm year derived from available columns
      - category_{...}: dummy variables created from original 'category' column (drop_first=True)

    Notes:
    - The function is defensive: it will attempt to coerce plausible alternative source
      columns into the required final columns when present (e.g., year -> StormYear).
    - It leaves rows intact where possible; the model function is responsible for
      handling missing endogenous observations.
    """
    df = df.copy()

    # Ensure relevant columns exist and coerce to numeric where appropriate
    # For conceptual columns listed in the contract we keep the exact names.
    for col in ['ndam15', 'name', 'elapsedyrs', 'wind', 'min', 'ind']:
        if col in df.columns:
            # keep original column name but coerce to numeric where appropriate
            # Note: 'name' is expected to be a numeric femininity rating; if it's not numeric,
            # this will coerce non-numeric values to NaN. We attempt to fill from plausible
            # alternate columns below if available.
            if col in ['ndam15', 'name', 'elapsedyrs', 'wind', 'min', 'ind']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            # Create as NA if missing so downstream code can handle gracefully
            df[col] = np.nan

    # Derive StormYear: prefer existing 'StormYear', else try 'year' or 'season'
    if 'StormYear' in df.columns:
        df['StormYear'] = pd.to_numeric(df['StormYear'], errors='coerce')
    elif 'year' in df.columns:
        df['StormYear'] = pd.to_numeric(df['year'], errors='coerce')
    elif 'season' in df.columns:
        df['StormYear'] = pd.to_numeric(df['season'], errors='coerce')
    else:
        df['StormYear'] = np.nan

    # If 'name' feminine rating is missing but there's a plausible alternate column, try to fill it.
    # (This is defensive; we do not change final column name 'name'.)
    if df['name'].isna().all():
        # common alternate names that might contain a numeric rating
        for alt in ['name_score', 'femininity', 'name_femininity', 'fem_score']:
            if alt in df.columns:
                df['name'] = pd.to_numeric(df[alt], errors='coerce')
                break

    # Binary indicator for female name from elapsedyrs (schema: 0 male, 1 female)
    # If elapsedyrs is not strictly 0/1, create NA for other values.
    def map_is_female(x):
        if pd.isna(x):
            return np.nan
        try:
            xi = int(x)
            if xi == 1:
                return 1
            if xi == 0:
                return 0
        except Exception:
            pass
        return np.nan

    df['is_female_name'] = df['elapsedyrs'].apply(map_is_female)

    # Mean-center the continuous name femininity score to improve interpretability
    # Keep the raw 'name' column as well
    if df['name'].notna().any():
        df['name_c'] = df['name'] - df['name'].mean()
    else:
        df['name_c'] = np.nan

    # Create dummy variables for 'category' (source of data). drop_first to avoid collinearity.
    if 'category' in df.columns:
        cat_dummies = pd.get_dummies(df['category'].astype(str), prefix='category', drop_first=True)
        # Concatenate dummies into dataframe
        df = pd.concat([df, cat_dummies], axis=1)

    # Do not aggressively drop rows here. The model function will handle missing endogenous values.
    # However, ensure returned dataframe contains the required final columns (even if NA).
    required_cols = [
        'ndam15', 'name', 'name_c', 'elapsedyrs', 'is_female_name',
        'wind', 'min', 'ind', 'StormYear'
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a Negative Binomial regression predicting hurricane fatalities (ndam15) from name femininity
    while controlling for storm severity and other covariates.

    Model specification:
      ndam15 ~ name_c + is_female_name + wind + min + ind + StormYear + category dummies

    Returns the fitted statsmodels results object (GLMResults). If there are no observations
    with non-missing ndam15, a small intercept-only fallback model will be fit to allow the
    pipeline to proceed; the returned results object will have attribute '_no_observations_fallback'
    set to True in that case.
    """
    # Prepare explanatory variables
    # Control columns (ensure they exist in df)
    control_cols = []
    for c in ['wind', 'min', 'ind', 'StormYear']:
        if c in df.columns:
            control_cols.append(c)

    # Category dummies created in transform have prefix 'category_'
    cat_cols = [c for c in df.columns if c.startswith('category_')]

    # Base independent variables
    iv_cols = []
    if 'name_c' in df.columns:
        iv_cols.append('name_c')
    if 'is_female_name' in df.columns:
        iv_cols.append('is_female_name')

    # Build model matrix
    model_cols = iv_cols + control_cols + cat_cols
    if len(model_cols) == 0:
        # create an empty exog frame (we'll add constant later)
        exog = pd.DataFrame(index=df.index)
    else:
        exog = df[model_cols].copy()
        # Coerce all exog columns to numeric and impute missing with 0 (explicit choice)
        for col in exog.columns:
            exog[col] = pd.to_numeric(exog[col], errors='coerce').fillna(0)

    # Endogenous variable
    endog = pd.to_numeric(df['ndam15'], errors='coerce')

    # Align rows: keep only rows with non-missing endog
    valid_mask = endog.notna()
    endog_valid = endog.loc[valid_mask]
    exog_valid = exog.loc[valid_mask]

    # If there are no valid observations (no non-missing ndam15), create a small intercept-only fallback dataset
    if endog_valid.shape[0] == 0:
        # Create a single-row intercept-only dataset as a fallback so the pipeline does not error.
        # This is explicitly marked on the returned results object.
        exog_fallback = pd.DataFrame({'const': [1.0]})
        endog_fallback = pd.Series([0.0])
        try:
            fam = sm.families.NegativeBinomial()
            glm_nb = sm.GLM(endog_fallback, exog_fallback, family=fam)
            results = glm_nb.fit()
        except Exception:
            glm_poi = sm.GLM(endog_fallback, exog_fallback, family=sm.families.Poisson())
            results = glm_poi.fit()
        setattr(results, "_no_observations_fallback", True)
        return results

    # Ensure exog_valid has columns; add constant (intercept)
    exog_valid = sm.add_constant(exog_valid, has_constant='add')

    # Final safety: convert exog to numeric array (statsmodels will handle dtype, but ensure no object dtypes)
    for col in exog_valid.columns:
        exog_valid[col] = pd.to_numeric(exog_valid[col], errors='coerce').fillna(0)

    # Fit Negative Binomial GLM
    try:
        fam = sm.families.NegativeBinomial()
        glm_nb = sm.GLM(endog_valid, exog_valid, family=fam)
        results = glm_nb.fit()
    except Exception:
        # If NegativeBinomial fails for any reason, fall back to Poisson
        glm_poi = sm.GLM(endog_valid, exog_valid, family=sm.families.Poisson())
        results_poi = glm_poi.fit()
        setattr(results_poi, "_fallback_to_poisson", True)
        results = results_poi

    return results