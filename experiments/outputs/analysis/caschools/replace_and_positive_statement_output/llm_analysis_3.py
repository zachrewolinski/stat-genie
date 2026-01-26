from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/caschools/replace_and_positive_statement_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Ensure numeric columns are numeric
    num_cols = ['students', 'teachers', 'computer', 'expenditure', 'income', 'english', 'lunch', 'calworks', 'read', 'math']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing the primary variables needed: students, teachers, read, math
    df = df.dropna(subset=['students', 'teachers', 'read', 'math'])

    # Compute dependent variable: average of reading and math scores
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Compute student-teacher ratio (students per teacher). Teachers should be > 0.
    # Use full-time-equivalent 'teachers' column
    df['student_teacher_ratio'] = df['students'] / df['teachers']

    # Remove implausible ratios (extreme outliers) and nonpositive
    df = df[df['student_teacher_ratio'] > 0]
    # Cap extremely large ratios (likely data or aggregation issues). Remove > 200 as implausible for this dataset.
    df = df[df['student_teacher_ratio'] < 200]

    # Derive computers per student to capture technology resources (guard against division by zero)
    df['computer_per_student'] = np.where(df['students'] > 0, df['computer'] / df['students'], np.nan)

    # Log transform of students to capture district size nonlinearity
    df['log_students'] = np.log1p(df['students'])

    # Normalize / clean grade-span indicator: create binary indicator for KK-08 (1) vs KK-06 or other (0)
    # Handle missing values by treating as 0 (will be dropped later if necessary)
    df['grades'] = df['grades'].astype(str).fillna('')
    df['grades_KK08'] = df['grades'].apply(lambda x: 1 if 'KK-08' in x else 0)

    # Keep only columns that will be used in the modeling plus a few identifiers
    keep_cols = [
        'rownames', 'district', 'school', 'county', 'grades',
        'students', 'teachers', 'computer', 'expenditure', 'income', 'english', 'lunch', 'calworks',
        'read', 'math',
        'AvgScore', 'student_teacher_ratio', 'computer_per_student', 'log_students', 'grades_KK08'
    ]
    # Keep intersection with columns present in df
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    # Final drop: ensure no missing values in model columns (drop rows with missing in any of these)
    model_cols = ['AvgScore', 'student_teacher_ratio', 'expenditure', 'income', 'english', 'lunch', 'calworks', 'computer_per_student', 'log_students', 'grades_KK08']
    model_cols = [c for c in model_cols if c in df.columns]
    df = df.dropna(subset=model_cols)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    import statsmodels.api as sm
    from sklearn.preprocessing import StandardScaler

    # Ensure the dataframe is the transformed one (contains our engineered columns)
    required = ['AvgScore', 'student_teacher_ratio', 'expenditure', 'income', 'english', 'lunch', 'calworks', 'computer_per_student', 'log_students', 'grades_KK08']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Dataframe missing required columns for modeling: {missing}")

    # Define outcome and predictors
    y = df['AvgScore']
    X = df[['student_teacher_ratio', 'expenditure', 'income', 'english', 'lunch', 'calworks', 'computer_per_student', 'log_students', 'grades_KK08']]

    # Add constant
    X = sm.add_constant(X)

    # Primary model: OLS with robust (HC3) standard errors
    ols_model = sm.OLS(y, X).fit(cov_type='HC3')

    # Secondary: standardized coefficients to express effect size in SD units
    scaler = StandardScaler()
    # Standardize predictors (exclude constant) and outcome
    X_std = scaler.fit_transform(df[['student_teacher_ratio', 'expenditure', 'income', 'english', 'lunch', 'calworks', 'computer_per_student', 'log_students', 'grades_KK08']])
    X_std = sm.add_constant(X_std)
    y_std = (df['AvgScore'] - df['AvgScore'].mean()) / df['AvgScore'].std()
    ols_std_model = sm.OLS(y_std, X_std).fit(cov_type='HC3')

    # Compute and return a concise summary dictionary plus full fit objects
    results = {
        'model': ols_model,
        'std_model': ols_std_model,
        'coef_student_teacher_ratio': float(ols_model.params.get('student_teacher_ratio', np.nan)),
        'pvalue_student_teacher_ratio': float(ols_model.pvalues.get('student_teacher_ratio', np.nan)),
        'std_coef_student_teacher_ratio': float(ols_std_model.params[1]) if len(ols_std_model.params) > 1 else np.nan,
        'n_obs': int(ols_model.nobs),
        'rsquared': float(ols_model.rsquared),
        'rsquared_adj': float(ols_model.rsquared_adj)
    }

    # Print brief summary to console for quick inspection
    print(ols_model.summary())
    print('\nStandardized-coefficient model (dependent variable standardized):')
    print(ols_std_model.summary())

    return results


