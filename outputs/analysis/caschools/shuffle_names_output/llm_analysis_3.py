from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/caschools/shuffle_names_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset to produce the variables required for analysis.

    Steps taken:
    - Make a copy of input frame
    - Coerce relevant columns to numeric where appropriate (tolerant to mismatched descriptions)
    - Create enrollment and teachers variables using the columns most likely representing them
      (based on the dataset description: 'calworks' appears to be total enrollment; 'teachers' is teacher FTE).
    - Compute student-teacher ratio and its log transform
    - Construct AvgScore as mean of the two standardized-score-like columns 'grades' and 'rownames' (these fields
      have values in ranges typical for test scores in the provided schema). If one is missing, mean will use the available one.
    - Coerce control variables and compute final dataframe used by the model. Rows with missing critical values (ratio, AvgScore)
      are dropped.

    Returns a dataframe containing columns described in the conceptual variables.
    """
    df = df.copy()

    # Coerce to numeric where appropriate. Use errors='coerce' to produce NaN for malformed entries.
    cols_to_numeric = ['calworks', 'teachers', 'grades', 'rownames', 'expenditure', 'math', 'district', 'income', 'computer']
    for c in cols_to_numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Enrollment: use 'calworks' (described in schema as total enrollment) if present.
    if 'calworks' in df.columns:
        df['Enrollment'] = df['calworks']
    else:
        # fallback: try to use 'students' or 'read' if available
        if 'students' in df.columns:
            # 'students' may not be numeric in schema; coerce above covered it
            df['Enrollment'] = pd.to_numeric(df['students'], errors='coerce')
        else:
            df['Enrollment'] = np.nan

    # Teachers: use 'teachers' column (FTE teachers)
    if 'teachers' in df.columns:
        df['Teachers'] = df['teachers']
    else:
        df['Teachers'] = np.nan

    # Compute student-teacher ratio. Avoid division by zero.
    df['StudentTeacherRatio'] = np.where(
        (df['Teachers'].notna()) & (df['Teachers'] > 0),
        df['Enrollment'] / df['Teachers'],
        np.nan
    )

    # Log transform (log1p to handle very small ratios gracefully)
    df['LogStudentTeacherRatio'] = np.log1p(df['StudentTeacherRatio'])

    # Dependent variable: average of available score columns that represent standardized scores.
    # Based on the schema 'grades' and 'rownames' appear to be reading and math average scores in 600-700 range.
    score_cols = [c for c in ['grades', 'rownames'] if c in df.columns]
    if len(score_cols) == 0:
        # If those aren't available, try fallback columns typically named 'read' or 'math'
        fallback = [c for c in ['read', 'math'] if c in df.columns]
        score_cols = fallback

    if len(score_cols) == 0:
        # No plausible score columns found; create AvgScore as NaN so downstream code fails loudly
        df['AvgScore'] = np.nan
    else:
        df['AvgScore'] = df[score_cols].mean(axis=1)

    # Controls: canonical mappings (coerce already applied)
    # Expenditure per student: 'expenditure' column (schema ambiguous but many versions use this name)
    if 'expenditure' in df.columns:
        df['ExpenditurePerStudent'] = df['expenditure']
    elif 'read' in df.columns:
        # fallback if expenditure mislabeled; more likely not needed
        df['ExpenditurePerStudent'] = pd.to_numeric(df['read'], errors='coerce')
    else:
        df['ExpenditurePerStudent'] = np.nan

    # Percent free/reduced-price lunch proxy: the schema field 'math' was ambiguously described as percent qualifying for reduced-price lunch
    if 'math' in df.columns:
        df['PctFreeLunch'] = df['math']
    else:
        df['PctFreeLunch'] = np.nan

    # Percent English learners: use 'district' column per schema description
    if 'district' in df.columns:
        df['PctEnglishLearners'] = df['district']
    else:
        df['PctEnglishLearners'] = np.nan

    # District income or socioeconomic measure: 'income' column
    if 'income' in df.columns:
        df['DistrictIncome'] = df['income']
    else:
        df['DistrictIncome'] = np.nan

    # Keep only rows with required analytic variables
    required_cols = ['StudentTeacherRatio', 'AvgScore']
    df = df.dropna(subset=required_cols)

    # For modeling convenience keep the final set of columns explicitly
    final_cols = [
        'Enrollment', 'Teachers', 'StudentTeacherRatio', 'LogStudentTeacherRatio', 'AvgScore',
        'ExpenditurePerStudent', 'PctFreeLunch', 'PctEnglishLearners', 'DistrictIncome'
    ]

    # Ensure all final columns exist in df
    for c in final_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Return only the final dataframe columns (preserves index)
    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS regression to estimate the association between student-teacher ratio and average academic performance.

    Primary specification:
      AvgScore ~ LogStudentTeacherRatio + ExpenditurePerStudent + PctFreeLunch + PctEnglishLearners + DistrictIncome

    - Uses log of student-teacher ratio to reduce influence of skew and to model relative changes.
    - Adds a constant and reports robust (HC3) standard errors.

    Returns the fitted statsmodels results object.
    """
    # Copy to avoid side-effects
    df = df.copy()

    # Define outcome and predictors
    y = df['AvgScore']

    # Predictor list: prefer log ratio; if it is missing, can fall back to raw ratio
    predictors = ['LogStudentTeacherRatio', 'ExpenditurePerStudent', 'PctFreeLunch', 'PctEnglishLearners', 'DistrictIncome']

    # Keep only rows with no missing values in y or predictors
    X = df[predictors]
    data = pd.concat([y, X], axis=1).dropna()

    if data.shape[0] < 10:
        raise ValueError(f"Too few observations after dropping missing values: {data.shape[0]}")

    y_clean = data['AvgScore']
    X_clean = data[predictors]

    # Add constant
    X_clean = sm.add_constant(X_clean, has_constant='add')

    # Fit OLS with robust standard errors (HC3)
    model_fit = sm.OLS(y_clean, X_clean).fit(cov_type='HC3')

    # Print a concise summary to stdout (optional). Return the fitted results object for programmatic use.
    try:
        print(model_fit.summary())
    except Exception:
        # If printing fails in certain execution environments, ignore
        pass

    return model_fit


