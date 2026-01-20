from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/affairs/shuffle_names_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy to avoid side-effects
    df = df.copy()

    # 1) Dependent variable: Affair frequency
    # According to the provided (messy) schema, the column 'education' actually contains the affair-frequency coding.
    # Convert to numeric and name it 'AffairFreq'.
    if 'education' in df.columns:
        df['AffairFreq'] = pd.to_numeric(df['education'], errors='coerce')
    else:
        # If column is missing, create AffairFreq as NaN to fail later if necessary
        df['AffairFreq'] = np.nan

    # 2) Independent variable: HasChildren (binary)
    # In this dataset schema the 'age' column contains 'yes'/'no' indicating presence of children.
    if 'age' in df.columns:
        # Normalize textual values, handle numeric encodings as fallback
        s = df['age']
        if s.dtype == object:
            mapped = s.astype(str).str.strip().str.lower().map({
                'yes': 1, 'y': 1, 'true': 1, 't': 1, '1': 1,
                'no': 0, 'n': 0, 'false': 0, 'f': 0, '0': 0
            })
            # If mapping produced NaN for some values, try to coerce to numeric
            mapped = mapped.fillna(pd.to_numeric(s, errors='coerce'))
            df['HasChildren'] = mapped.astype(float)
        else:
            # numeric or boolean: treat nonzero as 1
            df['HasChildren'] = (pd.to_numeric(s, errors='coerce').fillna(0) != 0).astype(float)
    else:
        # fallback: try to infer from a column named 'children'
        if 'children' in df.columns:
            s = df['children']
            # if values are 'male'/'female' that's gender; we cannot infer children -> set NaN
            df['HasChildren'] = np.nan
        else:
            df['HasChildren'] = np.nan

    # 3) Controls: create and coerce to numeric where appropriate
    # Gender: in this schema the 'children' column contains 'male'/'female'
    if 'children' in df.columns:
        df['Gender'] = df['children'].astype(str).str.strip().str.lower().map({'male': 1, 'm': 1, 'female': 0, 'f': 0})
        # If some entries are NA after mapping, try numeric coercion
        df['Gender'] = df['Gender'].fillna(pd.to_numeric(df['children'], errors='coerce'))
    else:
        df['Gender'] = np.nan

    # AgeYears: the 'rating' column in the schema contains midpoint age codes (e.g., 17.5, 22, 27...)
    df['AgeYears'] = pd.to_numeric(df['rating'], errors='coerce') if 'rating' in df.columns else np.nan

    # YearsMarried: schema indicates 'gender' actually contains years married coding
    df['YearsMarried'] = pd.to_numeric(df['gender'], errors='coerce') if 'gender' in df.columns else np.nan

    # EducationLevel: schema indicates 'affairs' column actually contains education coding (9,12,14...)
    df['EducationLevel'] = pd.to_numeric(df['affairs'], errors='coerce') if 'affairs' in df.columns else np.nan

    # Religiousness
    df['Religiousness'] = pd.to_numeric(df['religiousness'], errors='coerce') if 'religiousness' in df.columns else np.nan

    # Marital satisfaction (rownames in schema)
    df['MaritalSatisfaction'] = pd.to_numeric(df['rownames'], errors='coerce') if 'rownames' in df.columns else np.nan

    # 4) Basic cleaning: drop rows missing the outcome or the main independent variable
    df = df.dropna(subset=['AffairFreq', 'HasChildren']).reset_index(drop=True)

    # 5) Ensure types are numeric floats for modeling
    numeric_cols = ['AffairFreq', 'HasChildren', 'Gender', 'AgeYears', 'YearsMarried', 'EducationLevel', 'Religiousness', 'MaritalSatisfaction']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # After coercion, drop any rows that lost the DV or IV due to coercion
    df = df.dropna(subset=['AffairFreq', 'HasChildren']).reset_index(drop=True)

    # Final DataFrame returned contains all columns required by the model
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS model predicting AffairFreq from HasChildren and controls.
    Returns the fitted statsmodels regression results object (with robust standard errors).

    Notes:
    - AffairFreq is coded as in the survey; it is an ordinal/count-like variable with many zeros.
      The classic analysis often used a tobit (censored) model. Here we present OLS with robust
      standard errors as a transparent baseline. If a Tobit is desired, a separate MLE estimation
      can be implemented.
    """
    # Work on a copy
    df = df.copy()

    # Define outcome and predictors. Use only columns present in the transformed dataframe.
    y = df['AffairFreq']

    predictors = ['HasChildren', 'Gender', 'AgeYears', 'YearsMarried', 'EducationLevel', 'Religiousness', 'MaritalSatisfaction']
    X = df[[c for c in predictors if c in df.columns]].copy()

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Fit OLS with robust (HC3) standard errors
    ols_model = sm.OLS(y, X, missing='drop').fit(cov_type='HC3')

    # Return the fitted model object (user can inspect summary via model.summary())
    return ols_model


