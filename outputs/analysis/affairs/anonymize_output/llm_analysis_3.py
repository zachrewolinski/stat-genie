from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm


def _find_first_series(df: pd.DataFrame, candidates):
    """Return the first Series found in df from the list of candidate column names (case-insensitive).
    If none found, return None.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return df[cand]
        # case-insensitive match
        cl = cand.lower()
        if cl in cols_lower:
            return df[cols_lower[cl]]
    return None


def _map_binary_from_mixed(series: pd.Series) -> pd.Series:
    """Map a mixed-type series to binary 1/0 where possible. Unknown -> np.nan."""
    if series is None:
        return None
    s = series.copy()

    def map_val(x):
        if pd.isna(x):
            return np.nan
        # Strings
        try:
            xs = str(x).strip().lower()
        except Exception:
            xs = None
        if xs is not None:
            if xs in {'yes', 'y', 'true', 't', '1', 'male', 'm'}:
                return 1
            if xs in {'no', 'n', 'false', 'f', '0', 'female', 'fem'}:
                return 0
        # Try numeric
        try:
            xn = float(x)
            if np.isnan(xn):
                return np.nan
            # If numeric and 0/1
            if xn == 1:
                return 1
            if xn == 0:
                return 0
            # Otherwise, treat >0 as 1, else 0
            return 1 if xn > 0 else 0
        except Exception:
            return np.nan

    return s.apply(map_val)


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into analysis-ready columns.

    Produces the following final columns (these exact names MUST exist in the returned dataframe):
      - AffairCount: numeric count-like variable (0 = none, ...)
      - AnyAffair: binary indicator (0/1) for AffairCount > 0
      - LogAffairCount: np.log1p(AffairCount) when AffairCount is present, else NaN
      - HasChildren: 1 if children present, 0 if not
      - IsMale: 1 if male, 0 if female
      - Age: numeric
      - YearsMarried: numeric
      - Religiosity: numeric
      - Education: numeric
      - Occupation: numeric
      - MaritalHappiness: numeric

    The function is robust to varying raw column names; it will attempt several common alternatives
    for each feature. Rows with missing values in any of the model-required final columns are dropped.
    """
    df = df.copy()

    # Identify source series for each conceptual variable using common candidate names
    s_affair = _find_first_series(df, ['feature2', 'affaircount', 'affair', 'affairs', 'AffairCount'])
    s_children = _find_first_series(df, ['feature6', 'haschildren', 'has_children', 'children', 'children_present', 'HasChildren'])
    s_gender = _find_first_series(df, ['feature3', 'gender', 'sex', 'IsMale', 'ismale'])
    s_age = _find_first_series(df, ['feature4', 'age', 'Age'])
    s_years_married = _find_first_series(df, ['feature5', 'yearsmarried', 'years_married', 'YearsMarried'])
    s_religiosity = _find_first_series(df, ['feature7', 'religiosity', 'religiousness', 'Religiosity'])
    s_education = _find_first_series(df, ['feature8', 'education', 'Education'])
    s_occupation = _find_first_series(df, ['feature9', 'occupation', 'Occupation'])
    s_marital_happiness = _find_first_series(df, ['feature10', 'maritalhappiness', 'marital_happiness', 'happiness', 'MaritalHappiness'])

    # Create final columns. If source is missing, fill with NaN (will be dropped later).
    # AffairCount: numeric
    if s_affair is not None:
        df['AffairCount'] = pd.to_numeric(s_affair, errors='coerce')
    else:
        df['AffairCount'] = np.nan

    # AnyAffair: 1 if AffairCount > 0
    df['AnyAffair'] = np.where(df['AffairCount'].notna(), (df['AffairCount'] > 0).astype(float), np.nan)

    # LogAffairCount: log1p if AffairCount is present, else NaN
    df['LogAffairCount'] = np.where(df['AffairCount'].notna(), np.log1p(df['AffairCount']), np.nan)

    # HasChildren: binary mapping
    mapped_children = _map_binary_from_mixed(s_children) if s_children is not None else None
    if mapped_children is not None:
        df['HasChildren'] = mapped_children.astype(float)
    else:
        # If final column already exists in input, try to use it directly
        if 'HasChildren' in df.columns:
            df['HasChildren'] = pd.to_numeric(df['HasChildren'], errors='coerce')
        else:
            df['HasChildren'] = np.nan

    # IsMale: binary mapping (male=1, female=0)
    mapped_gender = _map_binary_from_mixed(s_gender) if s_gender is not None else None
    if mapped_gender is not None:
        # The mapping may have mapped 'female'->0 and 'male'->1 already; numeric values handled similarly.
        # But if a source used 'male'/'female' strings, we want male=1, female=0 already covered.
        df['IsMale'] = mapped_gender.astype(float)
    else:
        if 'IsMale' in df.columns:
            df['IsMale'] = pd.to_numeric(df['IsMale'], errors='coerce')
        else:
            df['IsMale'] = np.nan

    # Controls: coerce to numeric; if source missing, produce NaN column
    df['Age'] = pd.to_numeric(s_age, errors='coerce') if s_age is not None else (pd.to_numeric(df['Age'], errors='coerce') if 'Age' in df.columns else np.nan)
    df['YearsMarried'] = pd.to_numeric(s_years_married, errors='coerce') if s_years_married is not None else (pd.to_numeric(df['YearsMarried'], errors='coerce') if 'YearsMarried' in df.columns else np.nan)
    df['Religiosity'] = pd.to_numeric(s_religiosity, errors='coerce') if s_religiosity is not None else (pd.to_numeric(df['Religiosity'], errors='coerce') if 'Religiosity' in df.columns else np.nan)
    df['Education'] = pd.to_numeric(s_education, errors='coerce') if s_education is not None else (pd.to_numeric(df['Education'], errors='coerce') if 'Education' in df.columns else np.nan)
    df['Occupation'] = pd.to_numeric(s_occupation, errors='coerce') if s_occupation is not None else (pd.to_numeric(df['Occupation'], errors='coerce') if 'Occupation' in df.columns else np.nan)
    df['MaritalHappiness'] = pd.to_numeric(s_marital_happiness, errors='coerce') if s_marital_happiness is not None else (pd.to_numeric(df['MaritalHappiness'], errors='coerce') if 'MaritalHappiness' in df.columns else np.nan)

    # Now drop rows with missing values in the final model-required columns
    model_cols = ['AffairCount', 'AnyAffair', 'HasChildren', 'IsMale', 'Age',
                  'YearsMarried', 'Religiosity', 'Education', 'Occupation', 'MaritalHappiness']
    # Ensure model_cols exist in df (they should by construction)
    missing_final_cols = [c for c in model_cols if c not in df.columns]
    if missing_final_cols:
        # If somehow any final column is missing, create it with NaN so dropna below works uniformly
        for c in missing_final_cols:
            df[c] = np.nan

    df = df.dropna(subset=model_cols)

    # Ensure integer types for binary variables (now that NaNs are removed)
    df['AnyAffair'] = df['AnyAffair'].astype(int)
    df['HasChildren'] = df['HasChildren'].astype(int)
    df['IsMale'] = df['IsMale'].astype(int)

    # Return dataframe containing original + new columns. The modeling functions will select needed columns.
    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit models to test whether having children decreases engagement in extramarital affairs.

    Returns a dictionary containing fitted model results objects or error messages.
    """
    results = {}
    covars = ['HasChildren', 'IsMale', 'Age', 'YearsMarried', 'Religiosity',
              'Education', 'Occupation', 'MaritalHappiness']

    # Check that required covariate columns exist
    missing = [c for c in covars if c not in df.columns]
    if missing:
        results['error'] = f"Missing covariate columns: {missing}"
        return results

    # Prepare design matrix with constant
    X = sm.add_constant(df[covars])

    # 1) Logistic regression for AnyAffair
    y_bin = df['AnyAffair']
    try:
        logit_model = sm.Logit(y_bin, X)
        logit_res = logit_model.fit(disp=False)
        results['logit_any_affair'] = logit_res
    except Exception as e:
        results['logit_any_affair_error'] = str(e)

    # 2) Zero-Inflated Poisson for AffairCount
    try:
        from statsmodels.discrete.count_model import ZeroInflatedPoisson
        y_count = df['AffairCount']
        zip_model = ZeroInflatedPoisson(endog=y_count, exog=X, exog_infl=X, inflation='logit')
        zip_res = zip_model.fit(disp=False, maxiter=100)
        results['zip_affaircount'] = zip_res
    except Exception as e:
        results['zip_affaircount_error'] = str(e)

    # 3) OLS on LogAffairCount among those with AffairCount > 0
    try:
        pos = df[df['AffairCount'] > 0].copy()
        if len(pos) >= 10:
            X_pos = sm.add_constant(pos[covars])
            y_log = pos['LogAffairCount']
            ols_model = sm.OLS(y_log, X_pos)
            ols_res = ols_model.fit()
            results['ols_log_positive_affaircount'] = ols_res
        else:
            results['ols_log_positive_affaircount_error'] = 'Too few positive cases to fit OLS (n < 10).'
    except Exception as e:
        results['ols_log_positive_affaircount_error'] = str(e)

    return results


if __name__ == "__main__":
    # Simple sanity check when run as a script (no I/O required).
    # If a CSV path is known, users can load it and call transform/model themselves.
    sample = pd.DataFrame({
        'feature2': [0, 1, 2, None],
        'feature3': ['male', 'female', 'male', 'female'],
        'feature4': [30, 27, 40, 35],
        'feature5': [5, 3, 10, 8],
        'feature6': ['yes', 'no', 'yes', 'no'],
        'feature7': [3, 2, 5, 4],
        'feature8': [2, 4, 3, 1],
        'feature9': [1, 2, 3, 4],
        'feature10': [4, 3, 5, 2]
    })
    transformed = transform(sample)
    print("Transformed dataframe:")
    print(transformed)
    res = model(transformed)
    print("Model keys:", list(res.keys()))