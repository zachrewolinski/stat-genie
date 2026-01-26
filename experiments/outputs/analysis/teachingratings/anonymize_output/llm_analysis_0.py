from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/teachingratings/anonymize_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw Hamermesh classroom dataset (feature1..feature13) into an analysis-ready dataframe.

    Produces these columns (at minimum) used in modeling:
      - Eval (dependent variable)
      - Beauty, Beauty_sq (independent variables)
      - Female, Age, Minority, SingleCourse, UpperDivision, NativeSpeaker, Tenure
      - NResponses, Enrollment, LogNResponses, LogEnrollment
      - InstructorID (for clustering)

    Notes:
      - Renames feature* columns to readable names.
      - Coerces numeric columns and drops rows missing Beauty, Eval, or InstructorID.
      - Creates binary dummies from categorical indicators.
    """
    df = df.copy()

    # Rename columns to meaningful names
    rename_map = {
        'feature1': 'RowID',
        'feature2': 'Minority',
        'feature3': 'Age',
        'feature4': 'Gender',
        'feature5': 'SingleCourse',
        'feature6': 'Beauty',
        'feature7': 'Eval',
        'feature8': 'Division',
        'feature9': 'NativeSpeaker',
        'feature10': 'Tenure',
        'feature11': 'NResponses',
        'feature12': 'Enrollment',
        'feature13': 'InstructorID'
    }
    df = df.rename(columns=rename_map)

    # Coerce to numeric where appropriate
    numeric_cols = ['Age', 'Beauty', 'Eval', 'NResponses', 'Enrollment', 'InstructorID']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows with missing key variables (dependent, main independent, clustering id)
    df = df.dropna(subset=['Beauty', 'Eval', 'InstructorID'])

    # Standardize textual categorical columns to lower-case strings to map reliably
    for c in ['Minority', 'Gender', 'SingleCourse', 'Division', 'NativeSpeaker', 'Tenure']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()

    # Create binary dummies
    # Gender: 'female' -> 1, else 0
    df['Female'] = df['Gender'].map(lambda x: 1 if x == 'female' else 0).fillna(0).astype(int)

    # Minority: 'yes' -> 1 else 0
    df['Minority'] = df['Minority'].map(lambda x: 1 if x == 'yes' else 0).fillna(0).astype(int)

    # SingleCourse: 'single' -> 1 else 0
    df['SingleCourse'] = df['SingleCourse'].map(lambda x: 1 if x == 'single' else 0).fillna(0).astype(int)

    # UpperDivision: Division 'upper' -> 1 else 0
    df['UpperDivision'] = df['Division'].map(lambda x: 1 if x == 'upper' else 0).fillna(0).astype(int)

    # NativeSpeaker and Tenure: 'yes' -> 1 else 0
    df['NativeSpeaker'] = df['NativeSpeaker'].map(lambda x: 1 if x == 'yes' else 0).fillna(0).astype(int)
    df['Tenure'] = df['Tenure'].map(lambda x: 1 if x == 'yes' else 0).fillna(0).astype(int)

    # Log transforms for enrollment and number of responses (class size controls)
    # Enrollment and NResponses should be > 0; dataset's min suggests they are positive
    df['LogEnrollment'] = np.log(df['Enrollment'].replace({0: np.nan}))
    df['LogNResponses'] = np.log(df['NResponses'].replace({0: np.nan}))

    # Quadratic term for beauty and an interaction with female for moderation test
    df['Beauty_sq'] = df['Beauty'] ** 2
    df['Beauty_Female'] = df['Beauty'] * df['Female']

    # Optionally drop rows with problematic logs (if Enrollment or NResponses missing/zero)
    # We'll keep rows with missing logs but modelling code may drop them automatically

    # Ensure InstructorID is integer for clustering
    df['InstructorID'] = pd.to_numeric(df['InstructorID'], errors='coerce').astype('Int64')

    # Return transformed dataframe (keeps all original columns plus new ones)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Runs an OLS regression of student evaluations on instructor beauty and controls.

    Model specification (linear OLS with clustered standard errors at the instructor level):
      Eval ~ Beauty + Beauty_sq + Female + Age + Minority + SingleCourse + UpperDivision + NativeSpeaker + Tenure + LogEnrollment + LogNResponses + Beauty:Female

    We cluster standard errors by InstructorID to account for non-independence of observations from the same instructor.

    Returns the fitted statsmodels result object (RegressionResultsWrapper).
    Also prints the model summary.
    """
    import statsmodels.formula.api as smf

    # Defensive copy
    df = df.copy()

    # Drop rows with missing values in covariates used by the model
    required_cols = ['Eval', 'Beauty', 'Beauty_sq', 'Female', 'Age', 'Minority', 'SingleCourse', 'UpperDivision', 'NativeSpeaker', 'Tenure', 'LogEnrollment', 'LogNResponses', 'InstructorID']
    missing_req = [c for c in required_cols if c not in df.columns]
    if len(missing_req) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing_req}")

    model_df = df.dropna(subset=required_cols)

    # Specify formula
    formula = (
        'Eval ~ Beauty + Beauty_sq + Female + Age + Minority + SingleCourse + '
        'UpperDivision + NativeSpeaker + Tenure + LogEnrollment + LogNResponses + Beauty:Female'
    )

    # Fit OLS with clustering by InstructorID
    # cov_kwds groups expects the grouping array
    results = smf.ols(formula=formula, data=model_df).fit(cov_type='cluster', cov_kwds={'groups': model_df['InstructorID']})

    # Print summary for quick inspection
    print(results.summary())

    return results


