from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/caschools/anonymize_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transformations performed:
    - Rename relevant columns to descriptive names used in the model
    - Coerce numeric columns and handle invalid values
    - Compute student_teacher_ratio = enrollment / num_teachers (set to NaN where num_teachers <= 0)
    - Compute AvgScore as the mean of average reading and average math scores
    - Compute computers_per_100_students = (num_computers / enrollment) * 100 (safe for zero enrollment)
    - Drop rows missing key variables (student_teacher_ratio or AvgScore)

    Final dataframe contains columns named explicitly in the conceptual variables above.
    """

    # Make a shallow copy to avoid modifying the input
    df = df.copy()

    # Rename columns (map dataset features to descriptive names)
    rename_map = {
        'feature6': 'enrollment',                 # total enrollment
        'feature7': 'num_teachers',               # number of teachers (FTE)
        'feature11': 'expenditure_per_student',   # expenditure per student
        'feature8': 'pct_calworks',               # percent CalWorks
        'feature9': 'pct_reduced_lunch',          # percent reduced-price lunch
        'feature13': 'pct_english_learners',      # percent English learners
        'feature10': 'num_computers',             # number of computers
        'feature12': 'district_income_k',         # district average income (in $1,000)
        'feature14': 'avg_reading_score',         # average reading score
        'feature15': 'avg_math_score',            # average math score
        'feature4': 'county',                     # county (categorical)
        'feature5': 'grade_span'                  # grade span (categorical)
    }
    df = df.rename(columns=rename_map)

    # Ensure numeric columns are numeric
    numeric_cols = ['enrollment', 'num_teachers', 'expenditure_per_student', 'pct_calworks',
                    'pct_reduced_lunch', 'pct_english_learners', 'num_computers',
                    'district_income_k', 'avg_reading_score', 'avg_math_score']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Basic cleaning: treat non-positive teachers or enrollment as missing for ratio calculation
    df.loc[df['num_teachers'] <= 0, 'num_teachers'] = np.nan
    df.loc[df['enrollment'] <= 0, 'enrollment'] = np.nan

    # Derived variables
    # Student-teacher ratio (students per teacher)
    df['student_teacher_ratio'] = df['enrollment'] / df['num_teachers']

    # Average score: mean of reading and math
    df['AvgScore'] = df[['avg_reading_score', 'avg_math_score']].mean(axis=1)

    # Computers per 100 students (handle division safely)
    df['computers_per_100_students'] = np.where(
        df['enrollment'].notna() & (df['enrollment'] > 0),
        df['num_computers'] / df['enrollment'] * 100,
        np.nan
    )

    # Keep columns needed for modeling (explicitly listed so users can inspect them)
    keep_cols = [
        'student_teacher_ratio', 'AvgScore', 'expenditure_per_student', 'pct_calworks',
        'pct_reduced_lunch', 'pct_english_learners', 'computers_per_100_students',
        'district_income_k', 'enrollment', 'num_teachers', 'grade_span', 'county'
    ]

    # If some expected columns are missing from the input dataset, add them as NaN so the downstream code fails gracefully
    for col in keep_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Drop rows missing the dependent variable or the primary independent variable
    df = df.dropna(subset=['AvgScore', 'student_teacher_ratio'])

    # Optionally: drop rows with extreme/infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['student_teacher_ratio', 'AvgScore'])

    # Reset index
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS regression of AvgScore on student_teacher_ratio controlling for district characteristics.
    - Categorical variables (grade_span, county) are converted to dummies (drop_first=True).
    - Robust standard errors (HC3) are requested to protect against heteroskedasticity.

    Returns the fitted statsmodels RegressionResults object.
    """

    # Copy to avoid modifying original
    data = df.copy()

    # Define predictors and outcome
    y = data['AvgScore']

    predictors = [
        'student_teacher_ratio',
        'expenditure_per_student',
        'pct_calworks',
        'pct_reduced_lunch',
        'pct_english_learners',
        'computers_per_100_students',
        'district_income_k',
        'enrollment'
    ]

    # Ensure predictors exist
    for p in predictors:
        if p not in data.columns:
            data[p] = np.nan

    X = data[predictors]

    # Create dummies for categorical controls: grade_span and county
    # Use drop_first=True to avoid multicollinearity
    if 'grade_span' in data.columns:
        d_grade = pd.get_dummies(data['grade_span'].astype(str), prefix='grade_span', drop_first=True)
        X = pd.concat([X, d_grade], axis=1)

    if 'county' in data.columns:
        d_county = pd.get_dummies(data['county'].astype(str), prefix='county', drop_first=True)
        X = pd.concat([X, d_county], axis=1)

    # Drop rows with missing data in X or y for the regression
    reg_df = pd.concat([y, X], axis=1).dropna()
    y_clean = reg_df['AvgScore']
    X_clean = reg_df.drop(columns=['AvgScore'])

    # Add constant
    X_clean = sm.add_constant(X_clean)

    # Fit OLS with robust standard errors (HC3)
    ols_model = sm.OLS(y_clean, X_clean)
    results = ols_model.fit(cov_type='HC3')

    # Return the fitted results object (caller can inspect summary via results.summary())
    return results


