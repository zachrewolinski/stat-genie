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
    """
    Transform the raw dataset into a dataframe with the variables needed for modeling.

    New/derived columns produced (and used in the model):
      - STRatio: students / teachers
      - AvgScore: (read + math) / 2
      - computer_per_student: computer / students
      - log_students: np.log1p(students)
      - grades_KK_08: binary indicator 1 if grades == 'KK-08', 0 otherwise

    The function drops rows with missing or invalid values in the core columns needed to compute these variables.
    """
    df = df.copy()

    # Ensure numeric columns are numeric (coerce errors to NaN)
    numeric_cols = ['students', 'teachers', 'read', 'math', 'computer', 'expenditure', 'income', 'english', 'lunch', 'calworks']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing the core outcome/predictor variables
    df = df.dropna(subset=['students', 'teachers', 'read', 'math'])

    # Remove rows with nonpositive teachers to avoid division by zero / nonsense STRatio
    df = df[df['teachers'] > 0]

    # Compute student-teacher ratio (students per teacher)
    df['STRatio'] = df['students'] / df['teachers']

    # Compute average score as mean of reading and math
    df['AvgScore'] = (df['read'] + df['math']) / 2.0

    # Computers per student
    # If computer or students missing, result will be NaN (kept for possible imputation or dropped by model)
    df['computer_per_student'] = df['computer'] / df['students']

    # Log of students to capture scale nonlinearity
    df['log_students'] = np.log1p(df['students'])

    # Binary indicator for grade-span KK-08 (reference: KK-06)
    # Handle cases where grades might be stored in different capitalization/spaces
    df['grades'] = df['grades'].astype(str).str.strip()
    df['grades_KK_08'] = df['grades'].apply(lambda x: 1 if x.upper().replace(' ', '') in ['KK-08', 'KK-8', 'KK-08'] else 0)

    # Keep only columns necessary for modeling plus identifying columns for reference
    model_columns = [
        'AvgScore', 'STRatio', 'expenditure', 'income', 'english', 'lunch', 'calworks',
        'computer_per_student', 'log_students', 'grades_KK_08', 'district', 'school', 'county'
    ]

    # Some control columns may not exist; filter list
    model_columns = [c for c in model_columns if c in df.columns]

    # Drop rows with missing values in the dependent variable or primary independent or core controls
    required_for_model = ['AvgScore', 'STRatio']
    for c in ['expenditure', 'income', 'english', 'lunch']:
        if c in df.columns:
            required_for_model.append(c)

    df = df.dropna(subset=[c for c in required_for_model if c in df.columns])

    # Return the dataframe (keeping relevant/model columns plus id fields)
    return df[model_columns + [c for c in ['rownames'] if c in df.columns]]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS regression to estimate the association between student-teacher ratio and average academic score.

    Model specification:
      AvgScore = beta0 + beta1 * STRatio + beta2 * expenditure + beta3 * income + beta4 * english
                 + beta5 * lunch + beta6 * calworks + beta7 * computer_per_student + beta8 * log_students
                 + beta9 * grades_KK_08 + error

    We use robust (HC1) standard errors and report coefficients and standard errors. We also compute VIFs
    to check multicollinearity among controls.
    """
    df = df.copy()

    # Define dependent and independent variables for the regression
    y = df['AvgScore']

    # Build design matrix with controls that exist in the dataframe
    predictors = ['STRatio', 'expenditure', 'income', 'english', 'lunch', 'calworks',
                  'computer_per_student', 'log_students', 'grades_KK_08']
    predictors = [p for p in predictors if p in df.columns]

    X = df[predictors]

    # Add constant
    X = sm.add_constant(X)

    # Fit OLS with robust standard errors (HC1)
    ols_model = sm.OLS(y, X).fit(cov_type='HC1')

    # Compute VIFs for predictors (excluding constant) to assess multicollinearity
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        vif_data = []
        X_for_vif = X.drop(columns=['const'], errors='ignore')
        # drop rows with any missing values in X_for_vif for VIF computation
        X_vif_clean = X_for_vif.dropna()
        for i, col in enumerate(X_vif_clean.columns):
            vif = variance_inflation_factor(X_vif_clean.values, i)
            vif_data.append({'variable': col, 'VIF': vif})
    except Exception:
        vif_data = None

    # Return a dictionary with the fitted model and diagnostics
    results = {
        'model': ols_model,
        'params': ols_model.params,
        'pvalues': ols_model.pvalues,
        'summary': ols_model.summary(),
        'vif': vif_data
    }

    return results


