from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/caschools/shuffle_names_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure numeric columns are numeric (coerce errors to NaN)
    df = df.copy()

    # Columns used to construct the student-teacher ratio
    # Based on the dataset schema, 'calworks' represents total enrollment and 'teachers' is teacher FTE
    df['calworks'] = pd.to_numeric(df['calworks'], errors='coerce')
    df['teachers'] = pd.to_numeric(df['teachers'], errors='coerce')

    # Dependent variable: 'grades' (average reading score). Ensure numeric
    df['grades'] = pd.to_numeric(df['grades'], errors='coerce')

    # Controls: expenditure (per-pupil), percent free/reduced lunch, percent English learners, number of computers
    # According to the provided schema descriptions: 'expenditure', 'math' (PctFreeLunch), 'district' (PctEnglishLearners), 'english' (Computers)
    df['expenditure'] = pd.to_numeric(df['expenditure'], errors='coerce')
    df['PctFreeLunch'] = pd.to_numeric(df['math'], errors='coerce')
    df['PctEnglishLearners'] = pd.to_numeric(df['district'], errors='coerce')
    df['Computers'] = pd.to_numeric(df['english'], errors='coerce')

    # Keep a categorical version of the school type
    df['school_type'] = df['school'].astype('category')

    # Remove impossible or missing teacher counts (can't divide by zero)
    df = df[df['teachers'].notna()]
    df = df[df['teachers'] > 0]

    # Drop rows missing the key variables needed for the analysis
    df = df.dropna(subset=['calworks', 'teachers', 'grades'])

    # Compute the student-teacher ratio (students per teacher)
    df['student_teacher_ratio'] = df['calworks'] / df['teachers']

    # Optionally generate a log-transformed ratio for robustness / linearization
    # Add a small constant to avoid log(0) if ever present (shouldn't be after filtering teachers>0)
    df['ln_student_teacher_ratio'] = np.log(df['student_teacher_ratio'] + 1e-6)

    # Standardize continuous controls for interpretability (mean=0, sd=1)
    for col in ['expenditure', 'PctFreeLunch', 'PctEnglishLearners', 'Computers']:
        if col in df.columns:
            df[col + '_z'] = (df[col] - df[col].mean()) / (df[col].std(ddof=0) if df[col].std(ddof=0) != 0 else 1)

    # Final dataframe includes the variables used in the model (keep extras for inspection)
    # Columns required by the model: grades, student_teacher_ratio, expenditure (or expenditure_z), PctFreeLunch, PctEnglishLearners, Computers, school_type
    # We'll keep both raw and z-scored controls; the model will use raw controls by default but z-scored columns are available.

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    import statsmodels.formula.api as smf

    # We'll use the transformed dataframe produced by transform().
    # Primary specification: OLS of grades on student_teacher_ratio controlling for district resources and demographics.

    # Use the raw continuous controls (you can substitute the _z standardized versions if preferred)
    formula = (
        'grades ~ student_teacher_ratio + expenditure + PctFreeLunch + '
        'PctEnglishLearners + Computers + C(school_type)'
    )

    # Fit OLS with robust standard errors (HC3)
    model_fit = smf.ols(formula=formula, data=df).fit(cov_type='HC3')

    # Return the fitted model object (statsmodels RegressionResults)
    return model_fit


