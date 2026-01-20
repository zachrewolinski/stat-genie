from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/teachingratings/noperturb_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw Hamermesh classroom dataset into analysis-ready dataframe.

    Output columns (these exact names are required by the statistical model):
      - Eval: dependent variable (from 'eval')
      - Beauty: independent variable (from 'beauty')
      - Beauty_sq: squared beauty term
      - Age: instructor age
      - Gender_Female: 1 if gender == 'female', 0 if 'male'
      - Minority: 1 if minority == 'yes', 0 if 'no'
      - Tenure: 1 if tenure == 'yes', 0 if 'no'
      - Native: 1 if native == 'yes', 0 if 'no'
      - Credits_Single: 1 if credits == 'single', 0 otherwise
      - Division_Upper: 1 if division == 'upper', 0 if 'lower'
      - LogStudents: natural log of 'students'
      - ProfID: professor identifier (from 'prof')

    Notes:
      - Rows with missing values on core variables (eval, beauty, students, prof) are dropped.
      - Beauty is left in its original scale (it is already mean-centered in the source data), and a quadratic term is added.
    """
    # Work on a copy
    df = df.copy()

    # Ensure required columns exist
    required_cols = ['eval', 'beauty', 'age', 'gender', 'minority', 'tenure', 'native', 'credits', 'division', 'students', 'prof']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Input dataframe is missing required columns: {missing}")

    # Drop rows with missing values in core modeling columns
    df = df.dropna(subset=['eval', 'beauty', 'students', 'prof'])

    # Dependent variable
    df['Eval'] = df['eval'].astype(float)

    # Independent variable(s)
    df['Beauty'] = df['beauty'].astype(float)
    df['Beauty_sq'] = df['Beauty'] ** 2

    # Controls
    # Age
    df['Age'] = pd.to_numeric(df['age'], errors='coerce')

    # Gender: create female indicator (1 if female, 0 if male). If other categories exist, treat as NaN.
    df['Gender_Female'] = df['gender'].map(lambda x: 1 if str(x).lower() == 'female' else (0 if str(x).lower() == 'male' else np.nan))

    # Minority, Tenure, Native: map 'yes'/'no' to 1/0
    df['Minority'] = df['minority'].map(lambda x: 1 if str(x).lower() == 'yes' else (0 if str(x).lower() == 'no' else np.nan))
    df['Tenure'] = df['tenure'].map(lambda x: 1 if str(x).lower() == 'yes' else (0 if str(x).lower() == 'no' else np.nan))
    df['Native'] = df['native'].map(lambda x: 1 if str(x).lower() == 'yes' else (0 if str(x).lower() == 'no' else np.nan))

    # Credits: single vs more
    df['Credits_Single'] = df['credits'].map(lambda x: 1 if str(x).lower() == 'single' else (0 if str(x).lower() == 'more' else np.nan))

    # Division: upper(1) vs lower(0)
    df['Division_Upper'] = df['division'].map(lambda x: 1 if str(x).lower() == 'upper' else (0 if str(x).lower() == 'lower' else np.nan))

    # Students: log transform to reduce skew
    # ensure numeric and positive
    df['Students'] = pd.to_numeric(df['students'], errors='coerce')
    df = df[df['Students'] > 0]
    df['LogStudents'] = np.log(df['Students'])

    # Professor identifier for clustering
    df['ProfID'] = pd.to_numeric(df['prof'], errors='coerce')

    # After generating derived columns, drop any rows with missing values in used model columns
    model_columns = ['Eval', 'Beauty', 'Beauty_sq', 'Age', 'Gender_Female', 'Minority', 'Tenure', 'Native', 'Credits_Single', 'Division_Upper', 'LogStudents', 'ProfID']
    df = df.dropna(subset=model_columns)

    # Keep only the columns needed for modeling (this also ensures column names match exactly what model expects)
    df = df[model_columns]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit OLS regression of Eval on Beauty (and squared term) with controls.

    Model specification:
      Eval ~ Beauty + Beauty_sq + Age + Gender_Female + Minority + Tenure + Native + Credits_Single + Division_Upper + LogStudents

    Standard errors are clustered at the professor (ProfID) level to account for multiple courses taught by the same instructor.

    Returns the fitted statsmodels results object (RegressionResultsWrapper).
    """
    # local import for formula interface
    import statsmodels.formula.api as smf

    # Validate that required columns exist
    required = ['Eval', 'Beauty', 'Beauty_sq', 'Age', 'Gender_Female', 'Minority', 'Tenure', 'Native', 'Credits_Single', 'Division_Upper', 'LogStudents', 'ProfID']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Transformed dataframe is missing required columns for modeling: {missing}")

    # Build formula
    formula = 'Eval ~ Beauty + Beauty_sq + Age + Gender_Female + Minority + Tenure + Native + Credits_Single + Division_Upper + LogStudents'

    # Fit OLS with clustered standard errors by professor
    # Some versions of statsmodels accept clustering via fit(cov_type='cluster', cov_kwds={'groups': df['ProfID']})
    # We use that interface here; if unavailable, the user can call .get_robustcov_results.
    model = smf.ols(formula=formula, data=df)
    results = model.fit(cov_type='cluster', cov_kwds={'groups': df['ProfID']})

    # Return the fitted results object so caller can inspect summary, params, conf_int, etc.
    return results


