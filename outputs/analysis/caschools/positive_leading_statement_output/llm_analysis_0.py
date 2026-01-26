from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/caschools/positive_leading_statement_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Ensure numeric columns are numeric
    numeric_cols = ['students', 'teachers', 'computer', 'expenditure', 'income', 'calworks', 'lunch', 'english', 'read', 'math']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with non-positive or missing students/teachers/read/math because they are essential
    required = ['students', 'teachers', 'read', 'math', 'expenditure', 'income', 'calworks', 'lunch', 'english', 'computer', 'grades', 'county']
    df = df.dropna(subset=required)

    # Remove rows with zero or negative teachers or students to avoid division by zero
    df = df[(df['teachers'] > 0) & (df['students'] > 0)]

    # Dependent variable: average of read and math
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Independent variable: student-teacher ratio (students per teacher)
    df['StuTeacherRatio'] = df['students'] / df['teachers']

    # Resource control: computers per student
    df['ComputersPerStudent'] = df['computer'] / df['students']

    # Cast categorical controls explicitly
    df['grades'] = df['grades'].astype('category')
    df['county'] = df['county'].astype('category')

    # Optional: winsorize extreme StuTeacherRatio values at 1st and 99th percentiles to reduce influence of outliers
    lower = df['StuTeacherRatio'].quantile(0.01)
    upper = df['StuTeacherRatio'].quantile(0.99)
    df['StuTeacherRatio'] = df['StuTeacherRatio'].clip(lower=lower, upper=upper)

    # Final drop any rows that still have missing values in model columns
    model_cols = ['AvgScore', 'StuTeacherRatio', 'expenditure', 'income', 'calworks', 'lunch', 'english', 'ComputersPerStudent', 'students', 'grades', 'county']
    df = df.dropna(subset=model_cols)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    import statsmodels.formula.api as smf

    # Formula: average score on student-teacher ratio and controls, county fixed effects
    # We expect a negative coefficient on StuTeacherRatio (higher ratio => lower performance).
    formula = (
        'AvgScore ~ StuTeacherRatio + expenditure + income + calworks + lunch + english '
        '+ ComputersPerStudent + students + C(grades) + C(county)'
    )

    # Fit OLS with robust (HC3) standard errors to account for heteroskedasticity
    model_fit = smf.ols(formula=formula, data=df).fit(cov_type='HC3')

    # Print a compact result summary (user-facing) and return the fitted model object
    print(model_fit.summary().tables[1])

    # Extract coefficient and p-value for the key independent variable
    coef = model_fit.params.get('StuTeacherRatio', None)
    pval = model_fit.pvalues.get('StuTeacherRatio', None)

    if coef is not None and pval is not None:
        direction = 'negative' if coef < 0 else 'positive'
        print(f"StuTeacherRatio coef = {coef:.4f} (two-sided p = {pval:.4f}); estimated effect is {direction}.")
        # One-sided p-value for hypothesis that lower ratio (smaller students/teacher) -> higher score
        # That corresponds to H1: coef < 0. One-sided p = p_two_sided / 2 when coef < 0
        one_sided = pval / 2.0 if coef < 0 else 1 - pval / 2.0
        print(f"One-sided p-value for coef < 0: {one_sided:.4f}")

    return model_fit


