from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/shuffle_names_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset to produce the columns used in the analysis/modeling.

    Notes on column mapping (based on the provided schema descriptions):
    - The original dataset labeling is inconsistent. The column named 'education' holds the coded
      extramarital frequency variable (0,1,2,3,7,12), so we map that to AffairFreq.
    - The original column named 'age' contains the yes/no indicator for children in the marriage,
      so we map that to HasChildren.
    - The original column named 'children' appears to contain gender categories (e.g., 'female','male'),
      so we map that to Female indicator.
    - The original column named 'rating' contains respondents' age coding and is mapped to Age.
    - The original column named 'gender' contains years married (numeric coded) and is mapped to YearsMarried.
    - The column named 'affairs' contains education level codes; we map to EducationLevel.
    """

    df = df.copy()

    # Helper to safely get a Series for a column name; if missing, return a NA series of correct length
    def _get_series(col_name: str) -> pd.Series:
        s = df.get(col_name)
        if s is None:
            return pd.Series([pd.NA] * len(df), index=df.index)
        return s

    # Create AffairFreq from the column that (per schema) contains extramarital frequency (named 'education')
    affair_series = pd.to_numeric(_get_series('education'), errors='coerce')
    df['AffairFreq'] = affair_series

    # Binary outcome: any affair in the past year
    # Build an Int64 (nullable integer) series explicitly to avoid object-dtype pitfalls
    any_affair = pd.Series(pd.NA, index=df.index, dtype='Int64')
    mask = affair_series.notna()
    if mask.any():
        # Convert boolean to plain ints first, then assign into nullable Int64 Series
        any_affair.loc[mask] = (affair_series.loc[mask] > 0).astype('int64')
    df['AnyAffair'] = any_affair

    # HasChildren: map the column that (per schema) contains yes/no about children (named 'age')
    s_age = _get_series('age').astype(str).str.strip().str.lower()
    def map_has_children(x):
        # x is a string here due to astype(str); handle common representations
        if pd.isna(x) or x == 'nan' or x == '<NA>':
            return pd.NA
        if x in {'yes', 'y', 'true', '1', 't'}:
            return 1
        if x in {'no', 'n', 'false', '0', 'f'}:
            return 0
        return pd.NA
    df['HasChildren'] = s_age.apply(map_has_children).astype('Int64')

    # Female indicator: derive from the column that (per schema) contains gender (named 'children')
    s_children = _get_series('children').astype(str).str.strip().str.lower()
    def map_female(x):
        if pd.isna(x) or x == 'nan' or x == '<NA>':
            return pd.NA
        if isinstance(x, str) and x.startswith('f'):
            return 1
        if isinstance(x, str) and x.startswith('m'):
            return 0
        return pd.NA
    df['Female'] = s_children.apply(map_female).astype('Int64')

    # Other controls (convert to numeric, coercing non-numeric to NaN)
    df['Age'] = pd.to_numeric(_get_series('rating'), errors='coerce')
    df['YearsMarried'] = pd.to_numeric(_get_series('gender'), errors='coerce')
    df['Religiousness'] = pd.to_numeric(_get_series('religiousness'), errors='coerce')
    df['EducationLevel'] = pd.to_numeric(_get_series('affairs'), errors='coerce')
    df['Occupation'] = pd.to_numeric(_get_series('occupation'), errors='coerce')
    df['MarriageSatisfaction'] = pd.to_numeric(_get_series('rownames'), errors='coerce')

    # Keep only rows with the key variables required for the main analysis
    required_for_model = [
        'AnyAffair', 'HasChildren', 'Age', 'YearsMarried', 'Religiousness',
        'EducationLevel', 'Female'
    ]

    # Drop rows with missing values in required variables
    df = df.dropna(subset=required_for_model)

    # Ensure proper dtypes (convert Int64 to plain int since we've dropped rows with NA in those cols)
    # Only convert if columns exist
    for col in ['AnyAffair', 'HasChildren', 'Female']:
        if col in df.columns:
            # Safe conversion because we've dropped rows with NA in required variables
            try:
                df[col] = df[col].astype(int)
            except Exception:
                # If conversion fails for some reason, keep as nullable Int64
                df[col] = df[col].astype('Int64')

    # Return transformed dataframe containing all columns used in modeling
    model_cols = [
        'AnyAffair', 'AffairFreq', 'HasChildren', 'Age', 'YearsMarried',
        'Religiousness', 'EducationLevel', 'Female', 'Occupation', 'MarriageSatisfaction'
    ]
    # If some model_cols are not present in df due to unexpected schema, include only existing ones
    model_cols = [c for c in model_cols if c in df.columns]

    return df[model_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """ 
    Fit a logistic regression of AnyAffair on HasChildren controlling for covariates.
    Also run an OLS (robust SEs) on AffairFreq as a robustness check.

    Returns a dictionary with keys 'logit' and 'ols' containing the fitted results objects.
    """

    # Ensure we operate on a copy
    df = df.copy()

    # Define dependent and independent variables for the logistic model
    dep = 'AnyAffair'
    iv = 'HasChildren'
    controls = ['Age', 'YearsMarried', 'Religiousness', 'EducationLevel', 'Female', 'Occupation', 'MarriageSatisfaction']

    # Keep only columns that exist in the dataframe
    cols = []
    if iv in df.columns:
        cols.append(iv)
    cols += [c for c in controls if c in df.columns]

    # Prepare default results
    logit_mod = None
    ols_results = None

    # Drop any rows with missing values in the chosen regressors/dependent
    if dep in df.columns and len(cols) > 0:
        model_df = df[[dep] + cols].dropna()
        if not model_df.empty:
            y = model_df[dep].astype(float)
            X = model_df[cols].astype(float) if len(cols) > 0 else pd.DataFrame(index=model_df.index)
            # Add constant
            X = sm.add_constant(X, has_constant='add')

            # Ensure there is at least one observation and exog is non-empty
            if y.size > 0 and X.size > 0 and X.shape[1] > 0:
                # Fit logistic regression (Logit). Use robust convergence settings.
                try:
                    logit_mod = sm.Logit(y, X).fit(disp=0)
                except Exception:
                    # If Logit fails to converge, try using GLM with binomial family
                    try:
                        logit_mod = sm.GLM(y, X, family=sm.families.Binomial()).fit()
                    except Exception:
                        logit_mod = None

    # Robustness: OLS on AffairFreq (treating coded frequency as numeric outcome)
    if 'AffairFreq' in df.columns and len(cols) > 0:
        ols_df = df[['AffairFreq'] + cols].dropna()
        if len(ols_df) > 0:
            y_ols = ols_df['AffairFreq'].astype(float)
            X_ols = ols_df[cols].astype(float)
            X_ols = sm.add_constant(X_ols, has_constant='add')
            if y_ols.size > 0 and X_ols.size > 0 and X_ols.shape[1] > 0:
                try:
                    ols_results = sm.OLS(y_ols, X_ols).fit(cov_type='HC3')
                except Exception:
                    ols_results = None

    return {
        'logit': logit_mod,
        'ols': ols_results
    }