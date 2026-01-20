from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/caschools/replace_with_rvs_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Ensure numeric columns are numeric
    numeric_cols = ['students', 'teachers', 'read', 'math', 'computer', 'expenditure', 'income', 'english', 'lunch', 'calworks']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing the core variables needed to compute IV and DV
    df = df.dropna(subset=['students', 'teachers', 'read', 'math'])

    # Compute dependent variable: average of reading and math scores
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Compute independent variable: students per teacher
    # Guard against division by zero
    df['teachers'] = df['teachers'].replace({0: np.nan})
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Remove extreme or invalid ratios: set impossible or missing ratios to NaN
    df.loc[~np.isfinite(df['StudentTeacherRatio']), 'StudentTeacherRatio'] = np.nan

    # Winsorize StudentTeacherRatio at 1st and 99th percentiles to reduce influence of extreme outliers
    if df['StudentTeacherRatio'].notna().sum() > 0:
        lower = df['StudentTeacherRatio'].quantile(0.01)
        upper = df['StudentTeacherRatio'].quantile(0.99)
        df['StudentTeacherRatio'] = df['StudentTeacherRatio'].clip(lower=lower, upper=upper)

    # Derived control: computers per student
    df['ComputersPerStudent'] = df['computer'] / df['students']
    df.loc[~np.isfinite(df['ComputersPerStudent']), 'ComputersPerStudent'] = np.nan

    # Grade-span indicator: 1 if district covers KK-08, 0 otherwise
    df['Grades_KK08'] = (df['grades'].astype(str) == 'KK-08').astype(int)

    # Keep the county column as-is for later one-hot encoding in the model
    # Drop rows that are missing critical controls (this reduces sample but avoids implicit imputation)
    # We keep county even if missing (get_dummies will handle it), but drop rows missing key controls
    df = df.dropna(subset=['StudentTeacherRatio', 'AvgScore', 'expenditure', 'income', 'english', 'lunch', 'calworks', 'ComputersPerStudent'])

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # Model the association between StudentTeacherRatio and AvgScore controlling for covariates
    # Prepare outcome and covariates
    # The transform step should have created the necessary columns; we still guard against missing data
    required_cols = ['AvgScore', 'StudentTeacherRatio', 'expenditure', 'income', 'english', 'lunch', 'calworks', 'ComputersPerStudent', 'Grades_KK08', 'county']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column {col} not found in dataframe")

    # Select baseline covariates
    control_vars = ['expenditure', 'income', 'english', 'lunch', 'calworks', 'ComputersPerStudent', 'Grades_KK08']

    X = df[['StudentTeacherRatio'] + control_vars].copy()
    y = df['AvgScore'].copy()

    # Encode county fixed effects as dummies (drop_first to avoid multicollinearity)
    county_dummies = pd.get_dummies(df['county'].astype(str), prefix='county', drop_first=True)
    if county_dummies.shape[1] > 0:
        X = pd.concat([X, county_dummies], axis=1)

    # Drop any remaining rows with missing values in X or y
    data = pd.concat([X, y], axis=1).dropna()
    X = data.drop(columns=['AvgScore'])
    y = data['AvgScore']

    # Add constant
    X = sm.add_constant(X)

    # Fit OLS with heteroskedasticity-robust (HC3) standard errors
    model = sm.OLS(y, X)
    results = model.fit(cov_type='HC3')

    # Print a short summary and return the fitted results object
    print(results.summary())
    return results


