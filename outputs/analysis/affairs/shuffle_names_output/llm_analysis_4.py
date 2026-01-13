from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/shuffle_names_output/affairs.csv')


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Psychology Today / Fair dataset into a cleaned dataframe ready for modeling.

    - Creates a binary HasChildren indicator from the 'age' column (which in this dataset encodes whether there are children in the marriage).
    - Creates the dependent variable AffairFreq from the 'education' column (which here encodes extramarital frequency).
    - Derives control variables from the columns that (due to schema misalignment) represent gender, age, years married, occupation, etc.
    - Converts types, handles common string encodings, and ensures the final DataFrame contains the exact columns required by the model.

    Returns the dataframe containing the exact columns referenced in the model.
    """

    df = df.copy()

    # Ensure source columns exist to avoid KeyErrors later; missing ones will be NaN
    source_cols = ['age', 'education', 'children', 'rating', 'gender', 'occupation', 'religiousness', 'affairs', 'rownames']
    for col in source_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Dependent variable: extramarital frequency encoded in 'education' for this dataset
    df['AffairFreq'] = pd.to_numeric(df['education'], errors='coerce')

    # Independent variable: whether there are children in the marriage.
    # The 'age' column (per provided schema) is actually the factor 'Are there children in the marriage?'
    # Handle common string values and some possible capitalizations.
    def map_has_children(x):
        if pd.isna(x):
            return np.nan
        # If it's already numeric-like
        if isinstance(x, (int, float, np.integer, np.floating)):
            try:
                xv = float(x)
                if xv in (0.0, 1.0):
                    return int(xv)
                # Heuristic: treat positive values >0 as yes, 0 as no
                if xv == 0.0:
                    return 0
                if xv > 0:
                    return 1
            except Exception:
                return np.nan
        # For strings, be permissive in parsing
        if isinstance(x, str):
            x_low = x.strip().lower()
            # direct matches
            if x_low in ['yes', 'y', '1', 'true', 't', 'have', 'has', 'children', 'child']:
                return 1
            if x_low in ['no', 'n', '0', 'false', 'f', 'none', 'no children', 'nochild']:
                return 0
            # check numeric tokens inside string
            for token in x_low.replace(',', ' ').split():
                if token.isdigit():
                    try:
                        tv = int(token)
                        if tv in (0, 1):
                            return tv
                        if tv > 1:
                            return 1
                    except Exception:
                        continue
            # check presence of y/n characters as fallback
            if 'yes' in x_low or (('y' in x_low) and ('no' not in x_low)):
                return 1
            if 'no' in x_low or 'n' in x_low:
                return 0
            return np.nan
        return np.nan

    df['HasChildren'] = df['age'].apply(map_has_children)

    # Control: gender. In this dataset 'children' column actually contains gender strings ('male'/'female')
    def map_is_male(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float, np.integer, np.floating)):
            try:
                xv = float(x)
                # Common encodings: 1 -> male, 2 -> female, 0 -> female/unknown
                if xv == 1.0:
                    return 1
                if xv in (2.0, 0.0):
                    return 0
                # fallback: treat positive odd as male, even as female
                if xv > 1:
                    return int(xv) % 2
                return 0
            except Exception:
                return np.nan
        if isinstance(x, str):
            xl = x.strip().lower()
            if xl in ['male', 'm', 'man', 'boy']:
                return 1
            if xl in ['female', 'f', 'woman', 'girl']:
                return 0
            # check tokens
            if 'male' in xl or ' man' in xl or xl.startswith('man'):
                return 1
            if 'female' in xl or ' woman' in xl or xl.startswith('woman'):
                return 0
            # numeric-like tokens
            for token in xl.replace(',', ' ').split():
                if token.isdigit():
                    try:
                        tv = int(token)
                        if tv == 1:
                            return 1
                        if tv == 2:
                            return 0
                    except Exception:
                        continue
            return np.nan
        return np.nan

    df['IsMale'] = df['children'].apply(map_is_male)

    # Controls: numeric conversions for other columns. Use provided columns but coerce errors to NaN.
    df['AgeYears'] = pd.to_numeric(df['rating'], errors='coerce')
    df['YearsMarried'] = pd.to_numeric(df['gender'], errors='coerce')
    df['Occupation'] = pd.to_numeric(df['occupation'], errors='coerce')
    df['Religiousness'] = pd.to_numeric(df['religiousness'], errors='coerce')
    # The 'affairs' column in the provided schema appears to contain education coding (9-20).
    df['EducationLevel'] = pd.to_numeric(df['affairs'], errors='coerce')
    df['MarriageHappiness'] = pd.to_numeric(df['rownames'], errors='coerce')

    # Keep only the exact columns required by the model (but do not drop rows here).
    model_cols = ['AffairFreq', 'HasChildren', 'IsMale', 'AgeYears', 'Religiousness', 'YearsMarried', 'Occupation', 'EducationLevel', 'MarriageHappiness']

    # Reindex to ensure all required columns are present in the final DataFrame (missing ones will be NaN)
    df = df.reindex(columns=model_cols)

    # Do not drop rows here; the model function will handle dropping or imputing observations as needed.
    # Cast binary columns to nullable integer dtype so NA is preserved but values are integer-like.
    # If mapping produced float-like 0.0/1.0, convert to integer where possible
    try:
        df['HasChildren'] = df['HasChildren'].astype('Int64')
    except Exception:
        df['HasChildren'] = pd.to_numeric(df['HasChildren'], errors='coerce').astype('Int64')
    try:
        df['IsMale'] = df['IsMale'].astype('Int64')
    except Exception:
        df['IsMale'] = pd.to_numeric(df['IsMale'], errors='coerce').astype('Int64')

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a linear regression model to estimate the relationship between having children and
    extramarital affair frequency, controlling for observed covariates.

    Returns the fitted statsmodels results object with robust (HC3) standard errors.
    """

    # Validate input contains required columns
    required_cols = ['AffairFreq', 'HasChildren', 'IsMale', 'AgeYears', 'Religiousness', 'YearsMarried', 'Occupation', 'EducationLevel', 'MarriageHappiness']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    if df.shape[0] == 0:
        raise ValueError("Input DataFrame contains no rows. Check transform() output for dropped rows or mapping issues.")

    # Prepare design matrix
    X_cols = ['HasChildren', 'IsMale', 'AgeYears', 'Religiousness', 'YearsMarried', 'Occupation', 'EducationLevel', 'MarriageHappiness']
    X = df[X_cols].copy()

    # Ensure numeric dtypes for exog
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # Dependent variable
    y = pd.to_numeric(df['AffairFreq'], errors='coerce')

    # Combine for row-wise operations
    combined = pd.concat([y, X], axis=1)

    # Require that the dependent variable and primary independent (HasChildren) are present.
    # Drop rows missing the dependent variable or the primary independent variable.
    combined = combined.dropna(subset=['AffairFreq', 'HasChildren'])

    if combined.shape[0] == 0:
        raise ValueError("No valid observations remain after requiring non-missing AffairFreq and HasChildren.")

    # For remaining missing control values, perform simple, transparent imputation:
    # - For binary controls (HasChildren, IsMale): fill with the column mode (most common value) if available, otherwise 0.
    # - For numeric controls: fill with the column median if available, otherwise 0.
    for col in X_cols:
        if col not in combined.columns:
            continue
        if combined[col].isna().all():
            # If entire column is missing, fill with 0 to allow the model to run.
            combined[col] = combined[col].fillna(0)
            continue

        if col in ['HasChildren', 'IsMale']:
            # Use mode if exists
            try:
                modes = combined[col].mode(dropna=True)
                if not modes.empty:
                    fill_val = int(modes.iloc[0])
                else:
                    fill_val = 0
            except Exception:
                fill_val = 0
            combined[col] = combined[col].fillna(fill_val).astype(float)
        else:
            # Numeric control: fill with median
            try:
                med = combined[col].median(skipna=True)
                if pd.isna(med):
                    med = 0.0
            except Exception:
                med = 0.0
            combined[col] = combined[col].fillna(med).astype(float)

    # Final y and X
    y = combined['AffairFreq'].astype(float)
    X = combined[X_cols].astype(float)

    # Add constant term
    X = sm.add_constant(X, has_constant='add')

    # Fit OLS with heteroskedasticity-robust SEs (HC3)
    ols_model = sm.OLS(y, X)
    results = ols_model.fit(cov_type='HC3')

    return results