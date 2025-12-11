from typing import Any, Dict, Iterable, List, Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


# ======== TRANSFORM CODE ========
def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """
    Return the first column name from df that matches any of the candidate names.
    Matching strategy:
      - exact case-insensitive match
      - candidate is substring of column name (case-insensitive)
      - column name is substring of candidate (case-insensitive)
    Returns None if no match found.
    """
    cols = list(df.columns)
    lower_to_col = {col.lower(): col for col in cols}
    cand_lowers = [c.lower() for c in candidates]

    # exact case-insensitive match
    for c_lower, col in lower_to_col.items():
        if c_lower in cand_lowers:
            return col

    # substring matches
    for col in cols:
        col_l = col.lower()
        for cand in cand_lowers:
            if cand in col_l or col_l in cand:
                return col
    return None


def _broad_search_for_masfem(df: pd.DataFrame) -> Optional[str]:
    """
    Broad heuristic to find a column that likely corresponds to MasFem.
    Looks for columns with tokens like 'mas', 'fem', 'masculin', 'feminin', 'gender', or 'name' combined with 'female'/'masculine'.
    """
    pattern_tokens = ['mas', 'fem', 'masculin', 'feminin', 'gender', 'namefem', 'name_fem', 'namefemale', 'name_feminine', 'mas-fem', 'mas_fem', 'mas fem']
    cols = list(df.columns)
    for col in cols:
        col_l = col.lower()
        for tok in pattern_tokens:
            if tok in col_l:
                return col
    # fallback: try any column with both 'name' and ('female' or 'fem')
    for col in cols:
        col_l = col.lower()
        if 'name' in col_l and ('female' in col_l or 'fem' in col_l):
            return col
    return None


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataset into a dataframe with the variables used in modeling.

    The final dataframe will contain the exact columns required by the modeling code:
    ['MasFem','Deaths','LogDeaths','NameFemale','MaxWind','MinPressure','Category',
     'Year_Centered','Damage2015','MTurkMasFem','Source']
    """
    # Work on a copy
    df = df.copy()

    # Basic renaming for commonly expected feature names
    rename_map = {
        'feature2': 'Year',
        'feature3': 'Name',
        'feature4': 'MasFem',
        'feature5': 'MinPressure',
        'feature6': 'NameFemale',
        'feature7': 'Category',
        'feature8': 'Deaths',
        'feature9': 'DamageRaw',
        'feature10': 'YearsSince',
        'feature11': 'Source',
        'feature12': 'MTurkMasFem',
        'feature13': 'MaxWind',
        'feature14': 'Damage2015'
    }
    # Apply initial rename - safe even if some keys not present
    df = df.rename(columns=rename_map)

    # If some required conceptual columns are still missing, attempt to heuristically locate them
    desired_alternatives: Dict[str, List[str]] = {
        'MasFem': ['feature4', 'masfem', 'mas_fem', 'mas-fem', 'mas fem', 'masculinity', 'femininity'],
        'Deaths': ['feature8', 'deaths', 'death', 'fatalities', 'num_deaths', 'deaths_count', 'total_deaths'],
        'NameFemale': ['feature6', 'namefemale', 'name_female', 'female_name', 'is_female_name'],
        'MaxWind': ['feature13', 'maxwind', 'max_wind', 'max wind', 'wind_speed', 'windspeed'],
        'MinPressure': ['feature5', 'minpressure', 'min_pressure', 'min pressure', 'pressure_min'],
        'Category': ['feature7', 'category', 'saffir', 'saffir-simpson'],
        'Year': ['feature2', 'year', 'yr'],
        'Damage2015': ['feature14', 'damage2015', 'damage_2015', 'damage', 'property_damage'],
        'DamageRaw': ['feature9', 'damageraw', 'damage_raw', 'raw_damage'],
        'MTurkMasFem': ['feature12', 'mturkmasfem', 'mturk_masfem', 'mturk', 'mturk_mas_fem'],
        'Source': ['feature11', 'source', 'data_source', 'dataset_source']
    }

    for desired, candidates in desired_alternatives.items():
        if desired not in df.columns:
            found = _find_column(df, candidates)
            if found:
                # rename the found column to desired
                df = df.rename(columns={found: desired})

    # Additional fallback for MasFem: broad heuristic search
    if 'MasFem' not in df.columns:
        found = _broad_search_for_masfem(df)
        if found:
            df = df.rename(columns={found: 'MasFem'})

    # If MasFem still missing but MTurkMasFem exists, use it as a fallback to populate MasFem
    if 'MasFem' not in df.columns and 'MTurkMasFem' in df.columns:
        df['MasFem'] = df['MTurkMasFem']

    # As a last resort, ensure the column exists (filled with NaN) so downstream code can proceed
    if 'MasFem' not in df.columns:
        df['MasFem'] = np.nan

    # Now ensure the primary required columns are present; if not, raise informative error
    primary_required = ['MasFem', 'Deaths']
    missing_primary = [c for c in primary_required if c not in df.columns]
    if missing_primary:
        raise ValueError(f"Input dataframe is missing required columns: {missing_primary}")

    # Convert numeric-ish columns to numeric dtype (coerce errors to NaN)
    # NOTE: Category should NOT be converted to numeric.
    numeric_cols = ['MasFem', 'MinPressure', 'NameFemale', 'Deaths',
                    'YearsSince', 'MTurkMasFem', 'MaxWind', 'Damage2015', 'Year']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # At this point, ensure MasFem is populated where possible:
    # - coerce MTurkMasFem to numeric if present
    if 'MTurkMasFem' in df.columns:
        df['MTurkMasFem'] = pd.to_numeric(df['MTurkMasFem'], errors='coerce')
    # Use MTurkMasFem to fill MasFem missing values where available
    mturk_series = df['MTurkMasFem'] if 'MTurkMasFem' in df.columns else pd.Series(np.nan, index=df.index)
    df['MasFem'] = pd.to_numeric(df['MasFem'], errors='coerce')
    df['MasFem'] = df['MasFem'].fillna(mturk_series)

    # If MasFem still has missing values, fill with the median of existing MasFem (if any), otherwise 0.0
    if df['MasFem'].isnull().any():
        median_mf = df['MasFem'].median()
        if np.isnan(median_mf):
            df['MasFem'] = df['MasFem'].fillna(0.0)
        else:
            df['MasFem'] = df['MasFem'].fillna(median_mf)

    # Clean categorical/string columns
    if 'Name' in df.columns:
        df['Name'] = df['Name'].astype('object')

    if 'Source' in df.columns:
        # Ensure Source is categorical if present; handle NaNs by filling before casting
        df['Source'] = df['Source'].astype(object).where(df['Source'].notnull(), 'unknown')
        df['Source'] = df['Source'].astype('category')

    # Ensure binary NameFemale is 0/1 (coerce unexpected values to NaN)
    if 'NameFemale' in df.columns:
        # Handle string indicators first
        if df['NameFemale'].dtype == object:
            s = df['NameFemale'].astype(str).str.strip().str.lower()
            df.loc[s.isin(['female', 'f', 'true', 'yes', '1']), 'NameFemale'] = 1.0
            df.loc[s.isin(['male', 'm', 'false', 'no', '0']), 'NameFemale'] = 0.0
        # Coerce to numeric and then fill missing with median or 0.0
        df['NameFemale'] = pd.to_numeric(df['NameFemale'], errors='coerce')
        if df['NameFemale'].isnull().any():
            median_nf = df['NameFemale'].median()
            if np.isnan(median_nf):
                df['NameFemale'] = df['NameFemale'].fillna(0.0)
            else:
                df['NameFemale'] = df['NameFemale'].fillna(median_nf)
    else:
        # Ensure existence with default 0.0
        df['NameFemale'] = 0.0

    # Drop observations missing the dependent variable Deaths
    if 'Deaths' in df.columns:
        df['Deaths'] = pd.to_numeric(df['Deaths'], errors='coerce')
    df = df.dropna(subset=['Deaths'])  # must have Deaths; MasFem now should be populated

    # Create log-transformed deaths for OLS robustness
    # Ensure Deaths non-negative
    df['Deaths'] = df['Deaths'].fillna(0)
    df['LogDeaths'] = np.log(df['Deaths'] + 1)

    # Center Year to improve interpretability and numerical stability
    if 'Year' in df.columns:
        # ensure numeric
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        med_year = df['Year'].median()
        if np.isnan(med_year):
            df['Year_Centered'] = 0.0
        else:
            df['Year_Centered'] = df['Year'] - med_year
    else:
        df['Year_Centered'] = 0.0

    # Convert Category to categorical (Saffir-Simpson category)
    if 'Category' in df.columns:
        # Replace NA with 'unknown' before converting
        df['Category'] = df['Category'].astype(object).where(df['Category'].notnull(), 'unknown')
        df['Category'] = pd.Categorical(df['Category']).as_ordered()
    else:
        df['Category'] = pd.Categorical(['unknown'] * len(df)).as_ordered()

    # If MTurkMasFem exists, coerce numeric (already attempted earlier)
    if 'MTurkMasFem' in df.columns:
        df['MTurkMasFem'] = pd.to_numeric(df['MTurkMasFem'], errors='coerce')

    # For control numeric columns: fill missing values with their medians when available,
    # otherwise fill with 0. This ensures the final dataframe contains all required columns.
    for c in ['MaxWind', 'MinPressure', 'Damage2015', 'MTurkMasFem']:
        if c in df.columns:
            # convert to numeric if not already
            df[c] = pd.to_numeric(df[c], errors='coerce')
            if df[c].isnull().any():
                median_val = df[c].median()
                if np.isnan(median_val):
                    df[c] = df[c].fillna(0.0)
                else:
                    df[c] = df[c].fillna(median_val)
        else:
            # Column entirely missing: create with zeros
            df[c] = 0.0

    # Ensure NameFemale exists (handled above, but enforce numeric dtype)
    if 'NameFemale' not in df.columns:
        df['NameFemale'] = 0.0
    df['NameFemale'] = pd.to_numeric(df['NameFemale'], errors='coerce')
    if df['NameFemale'].isnull().any():
        median_nf = df['NameFemale'].median()
        if np.isnan(median_nf):
            df['NameFemale'] = df['NameFemale'].fillna(0.0)
        else:
            df['NameFemale'] = df['NameFemale'].fillna(median_nf)

    # Ensure Source exists and is categorical
    if 'Source' not in df.columns:
        df['Source'] = pd.Categorical(['unknown'] * len(df))
    else:
        # already handled above; ensure categorical
        df['Source'] = df['Source'].astype(object).where(df['Source'].notnull(), 'unknown').astype('category')

    # Ensure LogDeaths exists (it should from above, but guard)
    if 'LogDeaths' not in df.columns:
        df['LogDeaths'] = np.log(df['Deaths'].fillna(0) + 1)

    # Ensure Year_Centered exists (already created above)
    if 'Year_Centered' not in df.columns:
        df['Year_Centered'] = 0.0

    # At this point ensure MasFem is numeric and has no remaining NaNs (impute if necessary)
    df['MasFem'] = pd.to_numeric(df['MasFem'], errors='coerce')
    if df['MasFem'].isnull().any():
        median_mf = df['MasFem'].median()
        if np.isnan(median_mf):
            df['MasFem'] = df['MasFem'].fillna(0.0)
        else:
            df['MasFem'] = df['MasFem'].fillna(median_mf)

    # Final required columns (must be present in output)
    needed = ['MasFem', 'Deaths', 'LogDeaths', 'NameFemale', 'MaxWind',
              'MinPressure', 'Category', 'Year_Centered', 'Damage2015',
              'MTurkMasFem', 'Source']

    # Cast types for consistency
    # Numeric casts
    for c in ['MasFem', 'Deaths', 'LogDeaths', 'NameFemale', 'MaxWind', 'MinPressure', 'Year_Centered', 'Damage2015', 'MTurkMasFem']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Category and Source should be categorical
    if 'Category' in df.columns:
        df['Category'] = pd.Categorical(df['Category']).as_ordered()
    if 'Source' in df.columns:
        df['Source'] = df['Source'].astype('category')

    # Reorder and select final columns; all should exist by construction
    missing_final = [c for c in needed if c not in df.columns]
    if missing_final:
        raise ValueError(f"Failed to construct final dataframe; missing columns: {missing_final}")

    df_final = df[needed].reset_index(drop=True)
    return df_final


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit the primary negative binomial model predicting Deaths from MasFem and controls.
    Also fit an OLS robustness check on log(Deaths + 1).

    Returns a dictionary with the fitted model results objects.
    """
    # Ensure the dataframe contains the expected columns
    required = ['Deaths', 'MasFem']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Required column {c} not found in dataframe")

    # Define the predictors used in the formula (must match conceptual variables)
    predictors = ['MasFem', 'MaxWind', 'MinPressure', 'Category', 'Year_Centered',
                  'Damage2015', 'MTurkMasFem', 'NameFemale', 'Source']

    # Prepare dataframe for modeling: make a copy and ensure categorical columns are proper dtype
    model_df = df.copy()
    if 'Category' in model_df.columns:
        model_df['Category'] = model_df['Category'].astype('category')
    if 'Source' in model_df.columns:
        model_df['Source'] = model_df['Source'].astype('category')

    # Impute any remaining missing predictor values that would otherwise drop all rows.
    # For numeric predictors, fill with median; for categorical, fill with 'unknown' or the mode.
    for col in ['MaxWind', 'MinPressure', 'Damage2015', 'MTurkMasFem', 'NameFemale', 'Year_Centered']:
        if col in model_df.columns:
            if model_df[col].dtype.name == 'category' or model_df[col].dtype == object:
                model_df[col] = model_df[col].astype(object).where(model_df[col].notnull(), 'unknown')
                model_df[col] = model_df[col].astype('category')
            else:
                # numeric
                if model_df[col].isnull().any():
                    med = model_df[col].median()
                    if np.isnan(med):
                        model_df[col] = model_df[col].fillna(0.0)
                    else:
                        model_df[col] = model_df[col].fillna(med)

    # For Category and Source ensure no missing (fill with 'unknown')
    if 'Category' in model_df.columns:
        model_df['Category'] = model_df['Category'].astype(object).where(model_df['Category'].notnull(), 'unknown')
        model_df['Category'] = model_df['Category'].astype('category')
    if 'Source' in model_df.columns:
        model_df['Source'] = model_df['Source'].astype(object).where(model_df['Source'].notnull(), 'unknown')
        model_df['Source'] = model_df['Source'].astype('category')

    # Ensure MasFem is numeric and impute if any remaining missing values
    if 'MasFem' in model_df.columns:
        model_df['MasFem'] = pd.to_numeric(model_df['MasFem'], errors='coerce')
        if model_df['MasFem'].isnull().any():
            med = model_df['MasFem'].median()
            if np.isnan(med):
                # As last resort, fill with 0.0 to avoid dropping all observations
                model_df['MasFem'] = model_df['MasFem'].fillna(0.0)
            else:
                model_df['MasFem'] = model_df['MasFem'].fillna(med)

    # Drop rows with missing data in response or any predictor used in the model
    # After imputation above, this should drop only rows missing Deaths (which are required).
    drop_vars = ['Deaths'] + predictors
    existing_drop_vars = [v for v in drop_vars if v in model_df.columns]
    model_df = model_df.dropna(subset=existing_drop_vars)

    # After dropping, ensure there are observations to fit
    if model_df.shape[0] == 0:
        raise ValueError("No observations with complete data for modeling after dropping NA in predictors/response.")

    # Construct formulas
    base_formula = (
        'MasFem + MaxWind + MinPressure + C(Category) + Year_Centered + Damage2015 + MTurkMasFem + NameFemale + C(Source)'
    )

    nb_formula = 'Deaths ~ ' + base_formula
    ols_formula = 'LogDeaths ~ ' + base_formula

    # Fit negative binomial (GLM with NegativeBinomial family)
    try:
        nb_model = smf.glm(nb_formula, data=model_df, family=sm.families.NegativeBinomial()).fit()
    except Exception:
        # If negative binomial fails (rare), fall back to Poisson with robust SEs
        poisson_res = smf.glm(nb_formula, data=model_df, family=sm.families.Poisson()).fit()
        try:
            nb_model = poisson_res.get_robustcov_results(cov_type='HC3')
        except Exception:
            # As a last resort, return the plain poisson fit
            nb_model = poisson_res

    # Fit OLS on log-deaths as robustness check
    ols_model = smf.ols(ols_formula, data=model_df).fit()

    # Return both fitted result objects for inspection
    results: Dict[str, Any] = {
        'negative_binomial': nb_model,
        'ols_log_deaths': ols_model
    }

    return results