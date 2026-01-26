from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/panda_nuts/anonymize_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Rename columns to meaningful names
    df = df.rename(columns={
        'feature1': 'IndividualID',
        'feature2': 'Age',
        'feature3': 'Sex',
        'feature4': 'HammerType',
        'feature5': 'NutsOpened',
        'feature6': 'SessionDuration',
        'feature7': 'HelpReceived'
    })

    # Drop rows missing core variables
    df = df.dropna(subset=['IndividualID', 'Age', 'Sex', 'NutsOpened', 'SessionDuration', 'HelpReceived'])

    # Ensure numeric columns
    df['NutsOpened'] = pd.to_numeric(df['NutsOpened'], errors='coerce')
    df['SessionDuration'] = pd.to_numeric(df['SessionDuration'], errors='coerce')
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

    # Drop rows with invalid durations or nuts
    df = df.dropna(subset=['NutsOpened', 'SessionDuration', 'Age'])
    df = df[df['SessionDuration'] > 0]

    # Compute efficiency: nuts opened per second
    df['Efficiency'] = df['NutsOpened'] / df['SessionDuration']

    # Encode Sex: male = 1, female = 0 (handle lowercase/uppercase)
    df['Sex_clean'] = df['Sex'].astype(str).str.strip().str.lower()
    df['Sex_Male'] = df['Sex_clean'].map({'m': 1, 'male': 1, 'f': 0, 'female': 0})
    # If mapping produced NaN (unexpected category), fill with 0 (conservative)
    df['Sex_Male'] = df['Sex_Male'].fillna(0).astype(int)
    df = df.drop(columns=['Sex_clean'])

    # Encode HelpReceived: yes = 1, no = 0
    df['Help_clean'] = df['HelpReceived'].astype(str).str.strip().str.lower()
    df['Help_Yes'] = df['Help_clean'].map({'y': 1, 'yes': 1, 'n': 0, 'no': 0})
    df['Help_Yes'] = df['Help_Yes'].fillna(0).astype(int)
    df = df.drop(columns=['Help_clean'])

    # Create explicit hammer-type dummies for the two non-reference types often present (Q, G).
    # Use 'wood' as the reference category (Hammer_Q=0 and Hammer_G=0 implies wood or other).
    df['HammerType'] = df['HammerType'].astype(str).str.strip()
    df['Hammer_Q'] = (df['HammerType'] == 'Q').astype(int)
    df['Hammer_G'] = (df['HammerType'] == 'G').astype(int)

    # Standardize age (z-score) for interpretability and scale stability
    age_mean = df['Age'].mean()
    age_std = df['Age'].std(ddof=0)
    if age_std == 0 or np.isnan(age_std):
        df['Age_z'] = 0.0
    else:
        df['Age_z'] = (df['Age'] - age_mean) / age_std

    # Ensure IndividualID is integer (grouping variable)
    try:
        df['IndividualID'] = df['IndividualID'].astype(int)
    except Exception:
        # if conversion fails, create a numeric code
        df['IndividualID'] = pd.factorize(df['IndividualID'])[0]

    # Final dataframe columns used in modeling
    model_cols = [
        'IndividualID', 'Age', 'Age_z', 'Sex_Male', 'Help_Yes',
        'Hammer_Q', 'Hammer_G', 'NutsOpened', 'SessionDuration', 'Efficiency'
    ]

    # Keep only available columns (defensive) and drop duplicates
    available = [c for c in model_cols if c in df.columns]
    df_out = df[available].copy()
    df_out = df_out.reset_index(drop=True)
    return df_out


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a mixed-effects linear model predicting Efficiency from Age_z, Sex_Male, Help_Yes,
    and hammer-type controls. Random intercept for IndividualID. If MixedLM fails,
    fall back to OLS with cluster-robust SE by IndividualID.

    Returns the fitted results object (MixedLMResults or RegressionResultsWrapper).
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Build the formula with the hammer controls (explicit columns created in transform)
    formula_terms = ['Age_z', 'Sex_Male', 'Help_Yes']
    # include hammer dummies if present
    hammer_cols = [c for c in ['Hammer_Q', 'Hammer_G'] if c in df.columns]
    formula_terms += hammer_cols
    formula = 'Efficiency ~ ' + ' + '.join(formula_terms)

    # Attempt mixed-effects model with random intercept for IndividualID
    try:
        md = smf.mixedlm(formula, df, groups=df['IndividualID'])
        mdf = md.fit(reml=False)
        results = mdf
    except Exception as e:
        # Fallback: ordinary least squares with cluster-robust SE by IndividualID
        ols = smf.ols(formula, df).fit()
        try:
            clustered = ols.get_robustcov_results(cov_type='cluster', groups=df['IndividualID'])
            results = clustered
        except Exception:
            # If clustering fails, return the plain OLS fit
            results = ols

    return results


