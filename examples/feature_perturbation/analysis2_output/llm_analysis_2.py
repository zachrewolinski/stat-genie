from typing import Any, Dict
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Note: the dataset read line from the original snippet is omitted here so this module can be imported
# and the transform function can be used with any DataFrame provided by the caller.


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe containing all variables used in modeling.
    Adds standardized versions of continuous predictors for easier interpretation.

    Input columns expected (per schema):
      - feature2: Year
      - feature4: masculinity-femininity index (1=masculine ... 11=feminine)
      - feature5: minimum pressure
      - feature6: binary name gender (0 male, 1 female)
      - feature7: Saffir-Simpson category
      - feature8: total number of deaths
      - feature13: maximum wind speed
      - feature14: damage adjusted to 2015

    Outputs (added columns used by model):
      - Fatalities, LogFatalities, MasFem, MasFem_z, FemaleName,
        MaxWind, MaxWind_z, MinPressure, MinPressure_z, Damage2015, Damage2015_z,
        Year, Year_z, Category
    """
    df = df.copy()

    # Map raw columns to analysis columns, coerce to numeric where appropriate
    df['Fatalities'] = pd.to_numeric(df.get('feature8'), errors='coerce')
    df['MasFem'] = pd.to_numeric(df.get('feature4'), errors='coerce')
    df['FemaleName'] = pd.to_numeric(df.get('feature6'), errors='coerce')
    df['MaxWind'] = pd.to_numeric(df.get('feature13'), errors='coerce')
    df['MinPressure'] = pd.to_numeric(df.get('feature5'), errors='coerce')
    df['Damage2015'] = pd.to_numeric(df.get('feature14'), errors='coerce')
    df['Year'] = pd.to_numeric(df.get('feature2'), errors='coerce')
    df['Category'] = pd.to_numeric(df.get('feature7'), errors='coerce')

    # For modeling, treat missing fatalities as zero (consistent with LogFatalities calculation below).
    # This avoids losing all observations when fatalities are simply not recorded as NaN.
    df['Fatalities'] = df['Fatalities'].fillna(0).astype(float)

    # Clip negative fatalities (if any) to zero to ensure valid counts
    df['Fatalities'] = df['Fatalities'].clip(lower=0.0)

    # Log transform for robustness (depends only on Fatalities)
    df['LogFatalities'] = np.log(df['Fatalities'] + 1)

    # Attempt to fill missing FemaleName where possible using MasFem (internal helpful imputation).
    # MasFem is on a 1 (masculine) ... 11 (feminine) scale. Use midpoint threshold 6: MasFem >=6 => female name.
    # This imputation is only applied to missing FemaleName values when MasFem is present.
    missing_female_mask = df['FemaleName'].isna() & df['MasFem'].notna()
    if missing_female_mask.any():
        df.loc[missing_female_mask, 'FemaleName'] = (df.loc[missing_female_mask, 'MasFem'] >= 6).astype(int)

    # If FemaleName still missing, default to 0 (male). This is a conservative fallback to ensure the final
    # dataframe contains the required binary column. We prefer imputation from MasFem when available above.
    if df['FemaleName'].isna().any():
        df['FemaleName'] = df['FemaleName'].fillna(0)

    # Impute missing Category where possible using MaxWind thresholds (Saffir-Simpson approximate cutoffs in mph).
    # This is an internal imputation to avoid dropping observations where category wasn't recorded but wind speed is available.
    cat_missing_mask = df['Category'].isna() & df['MaxWind'].notna()
    if cat_missing_mask.any():
        def _derive_category_from_wind(w):
            try:
                w = float(w)
            except Exception:
                return np.nan
            if w >= 157:
                return 5
            if w >= 130:
                return 4
            if w >= 111:
                return 3
            if w >= 96:
                return 2
            if w >= 74:
                return 1
            return 0  # below hurricane strength
        df.loc[cat_missing_mask, 'Category'] = df.loc[cat_missing_mask, 'MaxWind'].apply(_derive_category_from_wind)

    # If Category is still missing, fill with rounded median if available, otherwise 0.
    if df['Category'].isna().any():
        if df['Category'].notna().any():
            median_cat = int(round(df['Category'].median()))
            df['Category'] = df['Category'].fillna(median_cat)
        else:
            df['Category'] = df['Category'].fillna(0)

    # For core continuous predictors, impute missing values with medians (or sensible defaults when no data).
    # This keeps observations in the dataset and allows standardization to proceed.
    continuous_raw = {
        'MasFem': 6.0,         # midpoint of 1..11 when no data available
        'MaxWind': 0.0,        # no wind recorded -> assume 0
        'MinPressure': 1013.0, # average sea-level pressure as fallback
        'Damage2015': 0.0,     # no damage recorded -> assume 0
        'Year': 2000.0         # arbitrary default year if none present
    }
    for col, default in continuous_raw.items():
        if col in df.columns:
            if df[col].notna().any():
                med = df[col].median()
                df[col] = df[col].fillna(med)
            else:
                df[col] = df[col].fillna(default)

    # Now standardize continuous variables (z-scores) using population std (ddof=0).
    def _standardize(series: pd.Series, name: str) -> pd.Series:
        mean = series.mean()
        std = series.std(ddof=0)
        if pd.isna(mean) or pd.isna(std) or std == 0:
            # Center (subtract mean) and replace any remaining NaN with 0
            return (series - mean).fillna(0).astype(float)
        else:
            return ((series - mean) / std).astype(float)

    df['MasFem_z'] = _standardize(df['MasFem'], 'MasFem')
    df['MaxWind_z'] = _standardize(df['MaxWind'], 'MaxWind')
    df['MinPressure_z'] = _standardize(df['MinPressure'], 'MinPressure')
    df['Damage2015_z'] = _standardize(df['Damage2015'], 'Damage2015')
    df['Year_z'] = _standardize(df['Year'], 'Year')

    # Ensure binary FemaleName is numeric (0/1).
    df['FemaleName'] = pd.to_numeric(df['FemaleName'], errors='coerce').fillna(0)
    # Round and clip to ensure only 0/1 values
    df['FemaleName'] = df['FemaleName'].round().clip(lower=0, upper=1).astype(int)

    # Final safety: ensure Category is numeric and has no NA
    df['Category'] = pd.to_numeric(df['Category'], errors='coerce').fillna(0).astype(int)

    # Source (not used as a main predictor, but kept if needed)
    if 'feature11' in df.columns:
        df['Source'] = df['feature11'].astype('category')

    # Final reset index and return
    df = df.reset_index(drop=True)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Fit a negative binomial model on fatalities (count outcome) with name femininity as the key predictor,
    controlling for objective storm intensity and temporal trends. Also fit an OLS on log(Fatalities+1)
    as a robustness check.

    Returns a dictionary with the fitted negative-binomial results (robust cov) and OLS (robust cov).
    If the negative binomial fit fails, attempts reasonable fallbacks (discrete NB, Poisson).
    If no count-based model can be fit (e.g., all-zero counts), 'neg_binomial' will be None but OLS will
    still be returned if it can be fit.
    """
    # work on a copy
    data = df.copy()

    # Define predictors (must match columns created in transform)
    predictors = [
        'MasFem_z',     # primary IV (standardized femininity)
        'FemaleName',   # binary name gender as a robustness control
        'MaxWind_z',
        'MinPressure_z',
        'Damage2015_z',
        'Year_z',
        'Category'      # numeric Saffir-Simpson category
    ]

    # Ensure predictors exist
    missing = [p for p in predictors if p not in data.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required predictor columns in transformed dataframe: {missing}")

    # Ensure dependent variables exist
    if 'Fatalities' not in data.columns or 'LogFatalities' not in data.columns:
        raise ValueError("Transformed dataframe must contain 'Fatalities' and 'LogFatalities' columns.")

    # Prepare a model dataframe containing only the required columns to avoid alignment issues
    required_for_model = predictors + ['Fatalities', 'LogFatalities']
    model_df = data[required_for_model].copy()

    # Impute or coerce types for predictors to avoid dropping excessive rows:
    # - For standardized continuous variables (z-scores), missing values correspond to mean; fill with 0.
    continuous_z = ['MasFem_z', 'MaxWind_z', 'MinPressure_z', 'Damage2015_z', 'Year_z']
    for col in continuous_z:
        if col in model_df.columns:
            model_df[col] = pd.to_numeric(model_df[col], errors='coerce').fillna(0.0).astype(float)

    # - Category: if still missing (unlikely because transform fills), set to 0
    if 'Category' in model_df.columns:
        model_df['Category'] = pd.to_numeric(model_df['Category'], errors='coerce').fillna(0).astype(float)

    # - FemaleName: must be present; coerce to numeric and drop rows where it's missing after coercion.
    model_df['FemaleName'] = pd.to_numeric(model_df['FemaleName'], errors='coerce')

    # Drop rows that have missing values in essential columns: Fatalities, LogFatalities, FemaleName.
    # Other predictors have been imputed above to avoid losing observations.
    model_df = model_df.dropna(subset=['Fatalities', 'LogFatalities', 'FemaleName']).reset_index(drop=True)

    if model_df.shape[0] == 0:
        raise ValueError(
            "No non-missing observations available for modeling after preparing design matrix. "
            "Check the input dataframe passed to model() (it must be the output of transform())."
        )

    # Prepare design matrix X and dependent variables
    # Ensure predictor columns are numeric and in the correct order
    X = model_df[predictors].astype(float)
    X = sm.add_constant(X, has_constant='add')

    # Prepare dependent variables; ensure counts are non-negative integers
    y_counts = model_df['Fatalities'].astype(float).clip(lower=0.0)
    # If fractional counts exist (unlikely), round to nearest integer for discrete models
    y_counts_int = np.round(y_counts).astype(int)

    y_log = model_df['LogFatalities'].astype(float)

    nb_res_robust = None

    # If all counts are zero, a count model will not be informative; skip fitting discrete count model in that case.
    if y_counts_int.sum() == 0:
        nb_res_robust = None
    else:
        # Fit models with error handling to provide clearer messages; attempt several options for robustness.
        last_exception = None
        try:
            # 1) Try GLM NegativeBinomial first
            nb_model = sm.GLM(y_counts_int, X, family=sm.families.NegativeBinomial())
            nb_res = nb_model.fit()
            nb_res_robust = nb_res.get_robustcov_results(cov_type='HC3')
        except Exception as e_glm:
            last_exception = e_glm
            try:
                # 2) Fallback to discrete NegativeBinomial (may be more stable)
                nb_disc = sm.NegativeBinomial(y_counts_int, X)
                nb_res = nb_disc.fit(disp=False)
                nb_res_robust = nb_res.get_robustcov_results(cov_type='HC3')
            except Exception as e_disc:
                last_exception = e_disc
                try:
                    # 3) Final fallback: Poisson GLM (robust to overdispersion via robust cov)
                    pois = sm.GLM(y_counts_int, X, family=sm.families.Poisson())
                    pois_res = pois.fit()
                    nb_res_robust = pois_res.get_robustcov_results(cov_type='HC3')
                except Exception as e_pois:
                    last_exception = e_pois
                    # If all attempts fail, raise a clear error summarizing attempts.
                    raise RuntimeError(
                        "Model fitting failed for negative-binomial (GLM), discrete negative-binomial, "
                        f"and Poisson fallback. Last error: {last_exception}"
                    )

    # Fit OLS on log(Fatalities + 1) as a robustness check
    try:
        ols_model = sm.OLS(y_log, X)
        ols_res = ols_model.fit()
        ols_res_robust = ols_res.get_robustcov_results(cov_type='HC3')
    except Exception as e:
        raise RuntimeError(f"OLS model fitting failed: {e}")

    # Return both fitted result objects (robustified). neg_binomial may be None if fitting was skipped (e.g., all-zero counts).
    return {
        'neg_binomial': nb_res_robust,
        'ols_log': ols_res_robust
    }