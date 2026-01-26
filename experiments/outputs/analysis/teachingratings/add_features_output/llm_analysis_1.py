from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/teachingratings/add_features_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the Hamermesh classroom dataset for modeling the effect of instructor beauty on evaluations.

    Produces the following columns used by the model:
      - Eval: dependent variable (from 'eval')
      - Beauty: mean-centered beauty (from 'beauty')
      - Beauty_sq: squared term of Beauty
      - Age: instructor age
      - Gender_male: dummy (1 if gender == 'male')
      - Minority_yes: dummy (1 if minority == 'yes')
      - Tenure_yes: dummy (1 if tenure == 'yes')
      - Native_yes: dummy (1 if native == 'yes')
      - Credits_single: dummy (1 if credits == 'single')
      - Division_upper: dummy (1 if division == 'upper')
      - Log_students: log(students) with small+ check
      - Prof: instructor id (from 'prof')

    Rows with missing key variables are dropped.
    """
    # work on a copy
    df = df.copy()

    # Drop observations with missing dv or iv or instructor id or students (necessary for log)
    df = df.dropna(subset=['eval', 'beauty', 'prof', 'students'])

    # Dependent variable (keep original name Eval for modeling)
    df['Eval'] = df['eval'].astype(float)

    # Mean-center beauty to improve interpretability and reduce collinearity with square
    df['Beauty'] = df['beauty'].astype(float) - df['beauty'].astype(float).mean()
    df['Beauty_sq'] = df['Beauty'] ** 2

    # Continuous controls
    df['Age'] = df['age'].astype(float)

    # Categorical -> dummies (explicit binary dummies for main categories seen in schema)
    df['Gender_male'] = (df['gender'].astype(str).str.lower() == 'male').astype(int)
    df['Minority_yes'] = (df['minority'].astype(str).str.lower() == 'yes').astype(int)
    df['Tenure_yes'] = (df['tenure'].astype(str).str.lower() == 'yes').astype(int)
    df['Native_yes'] = (df['native'].astype(str).str.lower() == 'yes').astype(int)
    df['Credits_single'] = (df['credits'].astype(str).str.lower() == 'single').astype(int)
    df['Division_upper'] = (df['division'].astype(str).str.lower() == 'upper').astype(int)

    # Log-transform students (number participating). Ensure positive values.
    df['students'] = pd.to_numeric(df['students'], errors='coerce')
    df = df[df['students'] > 0]
    df['Log_students'] = np.log(df['students'].astype(float))

    # Instructor id for clustering
    # create a named Prof column (capital P) to match model specification; keep integer type
    df['Prof'] = pd.to_numeric(df['prof'], errors='coerce').astype('Int64')

    # Drop any rows with NA in newly created model columns
    model_cols = ['Eval', 'Beauty', 'Beauty_sq', 'Age', 'Gender_male', 'Minority_yes', 'Tenure_yes', 'Native_yes', 'Credits_single', 'Division_upper', 'Log_students', 'Prof']
    df = df.dropna(subset=model_cols)

    # Convert Prof to plain int (no pandas NA) for grouping in clustering if necessary
    df['Prof'] = df['Prof'].astype(int)

    # Return dataframe containing at least the necessary columns (keeps original columns too)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Estimate the effect of instructor beauty on student evaluations using OLS with cluster-robust standard errors at the instructor level.

    Model specification:
      Eval ~ Beauty + Beauty_sq + Age + Gender_male + Minority_yes + Tenure_yes + Native_yes + Credits_single + Division_upper + Log_students

    Returns the fitted statsmodels regression results object (with cluster-robust SEs by Prof).
    """
    import statsmodels.formula.api as smf

    # Ensure df contains the columns expected by the transform step
    required = ['Eval', 'Beauty', 'Beauty_sq', 'Age', 'Gender_male', 'Minority_yes', 'Tenure_yes', 'Native_yes', 'Credits_single', 'Division_upper', 'Log_students', 'Prof']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Define formula
    formula = 'Eval ~ Beauty + Beauty_sq + Age + Gender_male + Minority_yes + Tenure_yes + Native_yes + Credits_single + Division_upper + Log_students'

    # Fit OLS
    ols_res = smf.ols(formula=formula, data=df).fit()

    # Compute cluster-robust covariance (clustered by Prof)
    # Use get_robustcov_results to attach cluster-robust cov
    ess = ols_res.get_robustcov_results(cov_type='cluster', groups=df['Prof'])

    # Print a brief summary to inspect
    print(ess.summary())

    # Return the robust-results object for programmatic inspection
    return ess