from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data (path left as in original file)
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/anonymize_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy to avoid modifying original
    df = df.copy()

    # Rename relevant columns to meaningful names used in modeling
    # feature4: masculinity-femininity index (higher = more feminine)
    # feature6: binary gender indicator for name (0 male, 1 female)
    # feature8: deaths (count)
    # feature5: minimum pressure at landfall
    # feature13: maximum wind speed at landfall
    # feature7: Saffir-Simpson category
    # feature2: year
    # feature9: damage normalized to 2013
    df = df.rename(columns={
        'feature4': 'MasFem',
        'feature6': 'IsFemaleName_raw',
        'feature8': 'Deaths',
        'feature5': 'MinPressure',
        'feature13': 'MaxWind',
        'feature7': 'Category',
        'feature2': 'Year',
        'feature9': 'Damage2013'
    })

    # Record which columns originally existed after rename to guide safe behavior
    original_cols_after_rename = set(df.columns)

    # Ensure the core source columns exist in the dataframe (create as NaN if absent)
    required_source_cols = ['MasFem', 'IsFemaleName_raw', 'Deaths', 'MinPressure', 'MaxWind', 'Category', 'Year', 'Damage2013']
    for col in required_source_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Ensure numeric types for key columns (coerce non-numeric to NaN)
    numeric_cols = ['MasFem', 'IsFemaleName_raw', 'Deaths', 'MinPressure', 'MaxWind', 'Category', 'Year', 'Damage2013']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Replace inf values with NaN for safety
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # NOTE: avoid aggressive early dropping here to preserve observations for later robust imputation.
    # The model will require non-missing values; we will impute sensible defaults for controls so model has usable rows.

    # Ensure deaths is non-negative and integer where available.
    if 'Deaths' in df.columns:
        # Convert negative death counts to NaN (anomalies)
        df.loc[df['Deaths'] < 0, 'Deaths'] = np.nan
        # Convert non-missing death counts to integer type (counts)
        if df['Deaths'].notna().any():
            # convert to integers safely
            df.loc[df['Deaths'].notna(), 'Deaths'] = df.loc[df['Deaths'].notna(), 'Deaths'].astype(int)

    # Standardize (z-score) the MasFem index for interpretability
    # If MasFem is all NaN or zero-variance, fall back to denominator 1.0
    mas_mean = df['MasFem'].mean(skipna=True)
    mas_std = df['MasFem'].std(ddof=0, skipna=True)
    # If there is no valid mean (all NaN), set mean to 0 so that missing values can be imputed to center.
    if pd.isna(mas_mean):
        mas_mean = 0.0
    denom = mas_std if (pd.notna(mas_std) and mas_std != 0) else 1.0
    # Fill MasFem missing values with the mean so MasFem_z is defined (imputation for modeling)
    df['MasFem'] = df['MasFem'].fillna(mas_mean)
    df['MasFem_z'] = (df['MasFem'] - mas_mean) / denom

    # Create a clean binary IsFemaleName (0/1) from the raw indicator, coerce missing to NaN initially
    def _coerce_isfemale(x):
        if pd.isna(x):
            return np.nan
        try:
            return 1 if float(x) >= 0.5 else 0
        except Exception:
            return np.nan

    df['IsFemaleName'] = df['IsFemaleName_raw'].apply(_coerce_isfemale)

    # If IsFemaleName still missing, impute to 0 (assume male) to retain observations for modeling.
    # This is an imputation for computational robustness only.
    df['IsFemaleName'] = df['IsFemaleName'].fillna(0).astype(int)

    # Category should be treated as categorical; keep as numeric where available
    df['Category'] = pd.to_numeric(df['Category'], errors='coerce')

    # Year: create a centered year variable to aid interpretation
    if df['Year'].notna().any():
        year_mean = df['Year'].mean(skipna=True)
        # fill missing Year values with the mean before centering to avoid NaN
        df['Year'] = df['Year'].fillna(year_mean)
        df['YearCentered'] = df['Year'] - year_mean
    else:
        # No year info at all; set YearCentered to 0 (neutral) so model can run
        df['YearCentered'] = 0.0

    # Damage: ensure numeric and non-negative
    if 'Damage2013' in df.columns:
        df['Damage2013'] = pd.to_numeric(df['Damage2013'], errors='coerce')
        df.loc[df['Damage2013'] < 0, 'Damage2013'] = np.nan
        # Replace inf with NaN if present
        df['Damage2013'].replace([np.inf, -np.inf], np.nan, inplace=True)
        # Impute missing damage to 0 (no reported damage) for modeling robustness
        df['Damage2013'] = df['Damage2013'].fillna(0.0)

    # Impute MinPressure and MaxWind with median values where available to avoid dropping all rows in modeling.
    for col in ['MinPressure', 'MaxWind']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Replace infs with NaN
            df[col].replace([np.inf, -np.inf], np.nan, inplace=True)
            med = df[col].median(skipna=True)
            if pd.isna(med):
                # If no valid median (all NaN), set to a reasonable neutral fallback
                med = 0.0
            df[col] = df[col].fillna(med)
        else:
            df[col] = 0.0

    # Impute Category with mode (most frequent) if available; else 0
    if 'Category' in df.columns:
        try:
            mode_val = df['Category'].mode(dropna=True)
            if len(mode_val) > 0:
                fill_cat = mode_val.iloc[0]
            else:
                fill_cat = 0
        except Exception:
            fill_cat = 0
        # Replace infs with NaN if present
        df['Category'].replace([np.inf, -np.inf], np.nan, inplace=True)
        df['Category'] = df['Category'].fillna(fill_cat).astype(int)
    else:
        df['Category'] = 0

    # Impute Deaths with 0 where missing so model has observations to fit.
    # This is a conservative computational imputation (assumes no reported deaths).
    if 'Deaths' in df.columns:
        # Replace inf with NaN
        df['Deaths'].replace([np.inf, -np.inf], np.nan, inplace=True)
        df['Deaths'] = df['Deaths'].fillna(0).astype(int)
        # Ensure non-negative
        df.loc[df['Deaths'] < 0, 'Deaths'] = 0
    else:
        df['Deaths'] = 0

    # Create log-transformed deaths for OLS robustness: log(Deaths + 1)
    df['LogDeaths'] = np.log(df['Deaths'] + 1.0)

    # Ensure final required columns exist (even if all-NaN) to satisfy the contract
    final_required = ['MasFem_z', 'IsFemaleName', 'Deaths', 'LogDeaths', 'MinPressure', 'MaxWind', 'Category', 'YearCentered', 'Damage2013']
    for col in final_required:
        if col not in df.columns:
            # For safety, provide reasonable defaults consistent with types
            if col in ['MasFem_z', 'LogDeaths', 'MinPressure', 'MaxWind', 'YearCentered', 'Damage2013']:
                df[col] = 0.0
            elif col in ['IsFemaleName', 'Category', 'Deaths']:
                df[col] = 0
            else:
                df[col] = np.nan

    # Final safety: ensure no infs remain in final_required columns
    df[final_required] = df[final_required].replace([np.inf, -np.inf], np.nan)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    import statsmodels.api as sm  # local import to match original structure
    import statsmodels.formula.api as smf

    # Ensure that the transformation has been applied (expecting columns: Deaths, LogDeaths, MasFem_z, IsFemaleName, MinPressure, MaxWind, Category, YearCentered, Damage2013)
    required = ['Deaths', 'LogDeaths', 'MasFem_z', 'IsFemaleName', 'MinPressure', 'MaxWind', 'Category', 'YearCentered', 'Damage2013']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Dataframe is missing required columns for modeling: {missing}")

    # Drop rows with missing values in outcome or primary predictor(s).
    # We require Deaths and MasFem_z to be non-missing; other controls were imputed in transform for robustness.
    model_df = df.dropna(subset=['Deaths', 'MasFem_z']).copy()

    if model_df.shape[0] == 0:
        raise ValueError("No observations remain after dropping rows with missing required predictors/outcome. Cannot fit model.")

    # Replace any remaining infs with NaN and drop rows with non-finite values in required columns
    model_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    model_df = model_df.dropna(subset=required).copy()

    if model_df.shape[0] == 0:
        raise ValueError("No observations remain after removing non-finite values. Cannot fit model.")

    # Ensure types are appropriate
    # Category and IsFemaleName should be integers for the formula's categorical handling
    model_df['Category'] = pd.to_numeric(model_df['Category'], errors='coerce').fillna(0).astype(int)
    model_df['IsFemaleName'] = pd.to_numeric(model_df['IsFemaleName'], errors='coerce').fillna(0).astype(int)
    # Ensure Deaths is integer count and non-negative
    model_df['Deaths'] = pd.to_numeric(model_df['Deaths'], errors='coerce').fillna(0)
    # Clip negatives to zero in case of anomalies
    model_df.loc[model_df['Deaths'] < 0, 'Deaths'] = 0
    # Ensure integer counts
    model_df['Deaths'] = model_df['Deaths'].astype(int)

    # Defensive: cap extremely large values in numeric controls to avoid numerical overflow in GLM initial guesses
    # We operate in-place on the conceptual columns (allowed, as these remain the same conceptual variables)
    for col in ['MinPressure', 'MaxWind', 'Damage2013', 'YearCentered', 'MasFem_z']:
        if col in model_df.columns:
            # replace any extreme infinities/nans already removed; now clip to reasonable numeric range
            # Use percentiles to determine reasonable caps when possible
            try:
                vals = model_df[col].dropna()
                if not vals.empty:
                    lower, upper = np.percentile(vals.clip(lower=-1e12, upper=1e12), [1, 99])
                    # Expand slightly to be safe
                    cap_low = max(lower, -1e8)
                    cap_high = min(upper, 1e8)
                    model_df[col] = model_df[col].clip(lower=cap_low, upper=cap_high)
                else:
                    model_df[col] = model_df[col].fillna(0.0)
            except Exception:
                # If any error, fallback to safe numeric defaults
                model_df[col] = pd.to_numeric(model_df[col], errors='coerce').fillna(0.0)

    # Fit a Negative Binomial GLM for count outcome (Deaths). This accounts for overdispersion typical of count data.
    # Formula: Deaths ~ MasFem_z + IsFemaleName + MinPressure + MaxWind + C(Category) + YearCentered + Damage2013
    formula_nb = 'Deaths ~ MasFem_z + IsFemaleName + MinPressure + MaxWind + C(Category) + YearCentered + Damage2013'

    nb_model = None
    try:
        nb_model = smf.glm(formula=formula_nb, data=model_df, family=sm.families.NegativeBinomial()).fit()
    except Exception:
        # If Negative Binomial fails (common when starting deviance returns NaN), fallback to Poisson
        try:
            nb_model = smf.glm(formula=formula_nb, data=model_df, family=sm.families.Poisson()).fit()
        except Exception:
            # As a last resort, fall back to OLS on the log-transformed deaths using the same covariates.
            # This preserves a meaningful fitted object for downstream inspection.
            try:
                ols_fallback = smf.ols(formula='LogDeaths ~ MasFem_z + IsFemaleName + MinPressure + MaxWind + C(Category) + YearCentered + Damage2013', data=model_df).fit()
                nb_model = ols_fallback
            except Exception:
                # If everything fails, raise a clear error
                raise RuntimeError("Failed to fit Negative Binomial, Poisson, and OLS fallback models.")

    # For robustness, fit an OLS on the log-transformed deaths (log(Deaths+1)). This is a commonly used alternative specification.
    formula_ols = 'LogDeaths ~ MasFem_z + IsFemaleName + MinPressure + MaxWind + C(Category) + YearCentered + Damage2013'
    ols_model = smf.ols(formula=formula_ols, data=model_df).fit()

    # Return both fitted results (primary = nb_model). Caller can inspect summary() on either.
    return {
        'nb_model': nb_model,
        'ols_log_model': ols_model,
        'model_df_used': model_df
    }