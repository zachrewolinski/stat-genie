from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/caschools/add_features_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw district-level dataframe to produce all columns needed for modeling.

    Produces the following new columns (these exact names are used in the model):
      - StudentTeacherRatio: students divided by teachers (NaN if teachers <= 0)
      - AvgScore: mean of 'read' and 'math' scores
      - ComputersPerStudent: computer / students
      - Expenditure: copied from 'expenditure'
      - Income: copied from 'income'
      - LunchPct: copied from 'lunch'
      - EnglishPct: copied from 'english'
      - Grades: factor version of 'grades'

    The function also coerces relevant columns to numeric and drops rows with missing values in the model variables.
    """
    df = df.copy()

    # Coerce important numeric columns to numeric (introduce NaN for non-numeric)
    numeric_cols = ['students', 'teachers', 'read', 'math', 'computer', 'expenditure', 'income', 'lunch', 'english']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Compute student-teacher ratio; avoid division by zero
    if 'students' in df.columns and 'teachers' in df.columns:
        df['StudentTeacherRatio'] = np.where(df['teachers'] > 0, df['students'] / df['teachers'], np.nan)
    else:
        df['StudentTeacherRatio'] = np.nan

    # Compute average score from read and math
    if 'read' in df.columns and 'math' in df.columns:
        df['AvgScore'] = df[['read', 'math']].mean(axis=1)
    else:
        df['AvgScore'] = np.nan

    # Computers per student (resource proxy)
    if 'computer' in df.columns and 'students' in df.columns:
        df['ComputersPerStudent'] = df['computer'] / df['students']
    else:
        df['ComputersPerStudent'] = np.nan

    # Copy / rename control columns to the exact names used in modeling
    if 'expenditure' in df.columns:
        df['Expenditure'] = df['expenditure']
    else:
        df['Expenditure'] = np.nan

    if 'income' in df.columns:
        df['Income'] = df['income']
    else:
        df['Income'] = np.nan

    if 'lunch' in df.columns:
        df['LunchPct'] = df['lunch']
    else:
        df['LunchPct'] = np.nan

    if 'english' in df.columns:
        df['EnglishPct'] = df['english']
    else:
        df['EnglishPct'] = np.nan

    # Grades as categorical factor (keep original labels)
    if 'grades' in df.columns:
        df['Grades'] = df['grades'].astype('category')
    else:
        df['Grades'] = pd.Categorical([None] * len(df))

    # Drop rows missing any of the variables we will use in the regression
    required_for_model = ['StudentTeacherRatio', 'AvgScore', 'Expenditure', 'Income', 'LunchPct', 'EnglishPct', 'ComputersPerStudent', 'Grades']
    df = df.dropna(subset=[c for c in required_for_model if c in df.columns])

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit an OLS model to estimate the association between student-teacher ratio and academic performance,
    controlling for district resources and demographics.

    Model formula (linear OLS):
      AvgScore ~ StudentTeacherRatio + Expenditure + Income + LunchPct + EnglishPct + ComputersPerStudent + C(Grades)

    Returns:
      - results: statsmodels RegressionResultsWrapper (fitted model)
    """
    # Import formula API locally to ensure availability
    import statsmodels.formula.api as smf

    # Ensure the required columns exist
    required = ['AvgScore', 'StudentTeacherRatio', 'Expenditure', 'Income', 'LunchPct', 'EnglishPct', 'ComputersPerStudent', 'Grades']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataframe is missing required columns for modeling: {missing}")

    # Specify formula; include Grades as a categorical factor using C(Grades)
    formula = 'AvgScore ~ StudentTeacherRatio + Expenditure + Income + LunchPct + EnglishPct + ComputersPerStudent + C(Grades)'

    model = smf.ols(formula=formula, data=df)

    # Fit with robust (HC3) standard errors to guard against heteroskedasticity
    results = model.fit(cov_type='HC3')

    # Print a brief summary for quick inspection
    print(results.summary())

    return results


