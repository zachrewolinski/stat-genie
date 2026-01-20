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
    # Work on a copy
    df = df.copy()

    # Keep rows with key variables present
    df = df.dropna(subset=['eval', 'beauty'])

    # Dependent variable: standardized column name
    df['Eval'] = pd.to_numeric(df['eval'], errors='coerce')

    # Independent variable: beauty (continuous). Keep raw and create z-score for interpretability
    df['Beauty'] = pd.to_numeric(df['beauty'], errors='coerce')
    # compute z using sample std (ddof=1) for interpretability
    df['Beauty_z'] = (df['Beauty'] - df['Beauty'].mean()) / df['Beauty'].std(ddof=1)

    # Controls: normalize and encode categorical variables to explicit binary columns
    # Gender: female = 1, male = 0
    df['Gender_Female'] = df['gender'].astype(str).str.lower().map({'female': 1, 'male': 0})

    # Minority: yes = 1, no = 0
    df['Minority_Yes'] = df['minority'].astype(str).str.lower().map({'yes': 1, 'no': 0})

    # Tenure: yes = 1, no = 0
    df['Tenure_Yes'] = df['tenure'].astype(str).str.lower().map({'yes': 1, 'no': 0})

    # Native English speaker: yes = 1, no = 0
    df['Native_Yes'] = df['native'].astype(str).str.lower().map({'yes': 1, 'no': 0})

    # Division: upper = 1, lower = 0
    df['Division_Upper'] = df['division'].astype(str).str.lower().map({'upper': 1, 'lower': 0})

    # Credits: single = 1, more = 0
    df['Credits_Single'] = df['credits'].astype(str).str.lower().map({'single': 1, 'more': 0})

    # Age and students should be numeric
    df['Age'] = pd.to_numeric(df['age'], errors='coerce')
    df['Students'] = pd.to_numeric(df['students'], errors='coerce')

    # Professor identifier (keep as-is for clustering or fixed effects)
    df['Prof'] = df['prof']

    # Fill any remaining binary mappings' NA with 0 (conservative) and cast to int
    for col in ['Gender_Female', 'Minority_Yes', 'Tenure_Yes', 'Native_Yes', 'Division_Upper', 'Credits_Single']:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # Interaction: beauty x gender (moderation test)
    df['Beauty_x_GenderFemale'] = df['Beauty_z'] * df['Gender_Female']

    # Define final model columns
    model_cols = [
        'Eval', 'Beauty', 'Beauty_z', 'Age', 'Gender_Female', 'Minority_Yes', 'Tenure_Yes',
        'Native_Yes', 'Division_Upper', 'Credits_Single', 'Students', 'Prof', 'Beauty_x_GenderFemale'
    ]

    # Drop rows with missing values in any model column
    df = df.dropna(subset=model_cols)

    # Return only the columns needed for modeling (preserves Prof for clustering/fixed effects)
    return df[model_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    import statsmodels.formula.api as smf

    # Formula: main effect of standardized beauty, controls, and interaction with gender
    formula = (
        'Eval ~ Beauty_z + Age + Gender_Female + Minority_Yes + Tenure_Yes + '
        'Native_Yes + Division_Upper + Credits_Single + Students + Beauty_x_GenderFemale'
    )

    # Fit OLS and compute cluster-robust SEs at the instructor level (Prof)
    ols_model = smf.ols(formula, data=df)
    results = ols_model.fit(cov_type='cluster', cov_kwds={'groups': df['Prof']})

    # Return the fitted results object (contains params, summary, robust SEs)
    return results


