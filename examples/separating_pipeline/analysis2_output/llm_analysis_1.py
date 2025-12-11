from typing import Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataset into a modeling-ready dataframe.

    The function:
    - Renames columns from the provided schema-style names (feature*) to descriptive names.
    - Coerces datatypes to numeric where appropriate.
    - Constructs the key variables described in the conceptual model:
        * Femininity: continuous masfem rating (higher = more feminine)
        * FemaleName: binary indicator (0 male / 1 female)
        * Deaths: raw death counts
    - Creates standardized (z-scored) versions of continuous control variables to aid interpretation and numerical stability in GLMs.
    - Creates a log(Deaths + 1) column for OLS robustness checks.
    - Ensures the final dataframe contains the required conceptual columns (possibly with NaNs if source data missing).
    """
    df = df.copy()

    # Rename columns to descriptive names used downstream
    rename_map = {
        'feature2': 'Year',
        'feature3': 'Name',
        'feature4': 'MasFem',        # continuous masculinity-femininity index (higher = more feminine per docs)
        'feature5': 'MinPressure',   # minimum central pressure at landfall
        'feature6': 'FemaleName',    # binary (0 male, 1 female)
        'feature7': 'Category',      # Saffir-Simpson category
        'feature8': 'Deaths',        # total number of deaths
        'feature9': 'Damage2013',    # damage normalized to 2013 dollars
        'feature10': 'YearsSince',
        'feature11': 'Source',
        'feature12': 'MTurkMasFem',
        'feature13': 'MaxWind',      # maximum wind speed at landfall
        'feature14': 'Damage2015'    # damage normalized to 2015 dollars
    }
    # apply renaming for only columns that exist
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Coerce key numeric columns to numeric types (safely)
    numeric_cols = [
        'MasFem', 'MTurkMasFem', 'MinPressure', 'FemaleName', 'Category', 'Deaths',
        'Damage2015', 'Damage2013', 'MaxWind', 'YearsSince', 'Year'
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Construct Femininity (final required column)
    # Prefer MasFem if available, otherwise fall back to MTurkMasFem, otherwise leave as NaN.
    fem_source_cols = ['MasFem', 'MTurkMasFem']
    fem_assigned = False
    for col in fem_source_cols:
        if col in df.columns and df[col].notna().any():
            df['Femininity'] = df[col].astype(float)
            fem_assigned = True
            break
    if not fem_assigned:
        # create Femininity as a column of NaNs so the final dataframe always contains it
        df['Femininity'] = np.nan

    # Ensure the binary variable is present and in a consistent numeric form
    if 'FemaleName' in df.columns:
        # force to numeric then to plain float dtype so patsy/statsmodels can handle missing values (np.nan)
        df['FemaleName'] = pd.to_numeric(df['FemaleName'], errors='coerce').round().astype(float)
    else:
        # create column filled with NA as float
        df['FemaleName'] = pd.Series([np.nan] * len(df), dtype=float)

    # Helper zscore function
    def zscore(col: pd.Series) -> pd.Series:
        # if all values are NA, return a series of NAs
        if col.notna().sum() == 0:
            return pd.Series([np.nan] * len(col), index=col.index, dtype=float)
        mean = col.mean()
        std = col.std(ddof=0)
        if pd.isna(std) or std == 0:
            std = 1.0
        return (col - mean) / std

    # Ensure raw control columns exist (create if missing) and create standardized versions
    control_raw_cols = ['MaxWind', 'MinPressure', 'Damage2015', 'Damage2013', 'YearsSince']
    for col in control_raw_cols:
        if col not in df.columns:
            df[col] = np.nan
        else:
            # ensure float dtype for consistency
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)

    # Damage2015_z: prefer Damage2015, fall back to Damage2013
    if df['Damage2015'].notna().any():
        df['Damage2015_z'] = zscore(df['Damage2015'])
    elif df['Damage2013'].notna().any():
        df['Damage2015_z'] = zscore(df['Damage2013'])
    else:
        df['Damage2015_z'] = pd.Series([np.nan] * len(df), index=df.index, dtype=float)

    # MaxWind_z and MinPressure_z
    df['MaxWind_z'] = zscore(df['MaxWind']) if df['MaxWind'].notna().any() else pd.Series([np.nan] * len(df), index=df.index, dtype=float)
    df['MinPressure_z'] = zscore(df['MinPressure']) if df['MinPressure'].notna().any() else pd.Series([np.nan] * len(df), index=df.index, dtype=float)

    # YearsSince_z
    df['YearsSince_z'] = zscore(df['YearsSince']) if df['YearsSince'].notna().any() else pd.Series([np.nan] * len(df), index=df.index, dtype=float)

    # Log transform of deaths for OLS robustness check (create Deaths if missing to keep contract)
    if 'Deaths' not in df.columns:
        df['Deaths'] = np.nan
    df['Deaths'] = pd.to_numeric(df['Deaths'], errors='coerce').astype(float)
    df['log_Deaths'] = np.log1p(df['Deaths'])

    # Ensure Year and Source exist in final dataframe
    if 'Year' not in df.columns:
        df['Year'] = np.nan
    else:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype(float)
    if 'Source' not in df.columns:
        df['Source'] = np.nan

    # Ensure Category exists and is numeric/categorical-compatible (use numeric with NaN allowed)
    if 'Category' not in df.columns:
        df['Category'] = np.nan
    else:
        df['Category'] = pd.to_numeric(df['Category'], errors='coerce')

    # Keep only columns required for modeling (and a few extras for diagnostics)
    keep_cols = [
        'Name', 'Year', 'Source',
        'Deaths', 'log_Deaths',
        'Femininity', 'FemaleName',
        'MaxWind', 'MaxWind_z', 'MinPressure', 'MinPressure_z',
        'Category', 'Damage2015', 'Damage2015_z', 'YearsSince', 'YearsSince_z'
    ]
    # Add any missing keep columns as NaN to preserve the contract
    for c in keep_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Return dataframe with columns in the specified order
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit the main statistical models to test whether more feminine hurricane names are associated
    with worse human outcomes (proxy: fatalities) after controlling for objective storm severity.

    Main model: Negative binomial GLM with Deaths as the count outcome. Negative binomial is chosen
    because Deaths are count data and often overdispersed relative to Poisson.

    Robustness: OLS on log(Deaths + 1) with the same covariates.

    Returns a dict containing the fitted models (nb_model, ols_model) so the caller can inspect
    summaries and coefficients. If there are no usable observations for fitting (after dropping
    missing values for variables in the model), returns models as None and includes a note.
    """
    import statsmodels.formula.api as smf

    # Ensure we have the columns we expect (presence only; rows with missing values will be handled by statsmodels)
    required = ['Deaths', 'Femininity', 'FemaleName']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found in dataframe. Run transform() first.")

    # Define candidate control terms, include only if column exists and has at least one non-missing value.
    control_terms = []
    control_cols = []
    if 'MaxWind_z' in df.columns and df['MaxWind_z'].notna().any():
        control_terms.append('MaxWind_z')
        control_cols.append('MaxWind_z')
    if 'MinPressure_z' in df.columns and df['MinPressure_z'].notna().any():
        control_terms.append('MinPressure_z')
        control_cols.append('MinPressure_z')
    if 'Damage2015_z' in df.columns and df['Damage2015_z'].notna().any():
        control_terms.append('Damage2015_z')
        control_cols.append('Damage2015_z')
    if 'YearsSince_z' in df.columns and df['YearsSince_z'].notna().any():
        control_terms.append('YearsSince_z')
        control_cols.append('YearsSince_z')
    # Category treated as categorical only if it has at least two distinct non-missing levels
    if 'Category' in df.columns:
        try:
            n_levels = int(df['Category'].nunique(dropna=True))
        except Exception:
            n_levels = 0
        if n_levels >= 2:
            control_terms.append('C(Category)')
            control_cols.append('Category')

    controls_str = ' + '.join(control_terms)

    # Build formulas
    if controls_str:
        formula_nb = f'Deaths ~ Femininity + FemaleName + {controls_str}'
        formula_ols = f'log_Deaths ~ Femininity + FemaleName + {controls_str}'
    else:
        formula_nb = 'Deaths ~ Femininity + FemaleName'
        formula_ols = 'log_Deaths ~ Femininity + FemaleName'

    # Prepare the DataFrame for modeling: require non-missing values for all variables used in formula
    # Identify the columns that will be used (endog + all raw predictor columns)
    used_cols = ['Deaths', 'Femininity', 'FemaleName'] + control_cols
    # Ensure used_cols are unique and present
    used_cols = [c for i, c in enumerate(used_cols) if c not in used_cols[:i]]
    # If any used column is missing from df (shouldn't be), add as NaN to ensure subset works predictably
    for c in used_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Drop rows with missing values in any of the used columns (statsmodels/patsy will do this as well,
    # but doing it explicitly lets us detect empty result before trying to fit models)
    model_df = df[used_cols].dropna()
    if model_df.shape[0] == 0:
        return {
            'nb_model': None,
            'ols_model': None,
            'note': 'No observations with complete data for model variables.'
        }

    # Fit Negative Binomial GLM on the subset dataframe
    try:
        nb_model = smf.glm(formula_nb, data=model_df, family=sm.families.NegativeBinomial()).fit()
    except Exception as e:
        # If NB fails, fall back to Poisson with robust standard errors as an alternative
        try:
            poisson = smf.glm(formula_nb, data=model_df, family=sm.families.Poisson()).fit()
            nb_model = poisson
            try:
                nb_model._fallback_reason = str(e)
            except Exception:
                pass
        except Exception as e2:
            # If fallback also fails, return with note
            return {
                'nb_model': None,
                'ols_model': None,
                'note': f'Both NegativeBinomial and Poisson fittings failed. NB error: {e}; Poisson error: {e2}'
            }

    # Fit OLS on log(Deaths + 1) as robustness
    try:
        ols_model = smf.ols(formula_ols, data=model_df).fit()
    except Exception as e:
        # If OLS fails, still return NB result and note
        return {
            'nb_model': nb_model,
            'ols_model': None,
            'note': f'OLS fitting failed: {e}'
        }

    # Return both fitted model objects so the caller can inspect .summary() etc.
    return {
        'nb_model': nb_model,
        'ols_model': ols_model
    }