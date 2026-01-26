from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/panda_nuts/positive_leading_statement_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset into analysis-ready dataframe.

    Produces the following columns (minimum) used in modeling:
      - age: numeric, unchanged
      - sex_male: binary (1 = male, 0 = female)
      - help_yes: binary (1 = received help, 0 = no help)
      - Efficiency_per_min: nuts opened per minute (nuts_opened * 60 / seconds)
      - LogEfficiency: log(Efficiency_per_min + small_constant)
      - hammer: categorical (tool type), NA filled with 'unknown'
      - chimpanzee: numeric ID

    Rows with missing critical values (nuts_opened, seconds, age, sex, help, chimpanzee)
    or non-positive session durations are removed.
    """
    df = df.copy()

    # Drop rows missing the primary measurement fields or with invalid session duration
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age', 'sex', 'help', 'chimpanzee'])
    # Ensure seconds positive
    df = df[df['seconds'] > 0]

    # Normalize and encode sex -> sex_male (1 if male, 0 if female)
    df['sex'] = df['sex'].astype(str).str.lower().str.strip()
    df['sex_male'] = df['sex'].map({'m': 1, 'male': 1, 'f': 0, 'female': 0})
    # If any remaining NaNs in sex_male (unexpected labels), drop those rows to avoid ambiguity
    df = df.dropna(subset=['sex_male'])
    df['sex_male'] = df['sex_male'].astype(int)

    # Normalize and encode help -> help_yes (1 if helped, 0 otherwise)
    df['help'] = df['help'].astype(str).str.lower().str.strip()
    df['help_yes'] = df['help'].map(lambda x: 1 if x in ['y', 'yes', 'true', '1'] else 0)
    df['help_yes'] = df['help_yes'].astype(int)

    # Ensure chimpanzee id numeric
    df['chimpanzee'] = pd.to_numeric(df['chimpanzee'], errors='coerce')
    df = df.dropna(subset=['chimpanzee'])
    df['chimpanzee'] = df['chimpanzee'].astype(int)

    # Hammer: keep as categorical control; fill missing with 'unknown'
    df['hammer'] = df['hammer'].astype(str).fillna('unknown').str.strip()

    # Compute efficiency as nuts per minute and log-transform (small constant to avoid log(0))
    df['Efficiency_per_min'] = df['nuts_opened'] * 60.0 / df['seconds']
    # Add a very small constant (1e-6) to avoid log(0); this keeps scale interpretable
    df['LogEfficiency'] = np.log(df['Efficiency_per_min'] + 1e-6)

    # Final defensive drop: remove rows with non-finite LogEfficiency
    df = df[np.isfinite(df['LogEfficiency'])]

    # Reset index for cleanliness
    df = df.reset_index(drop=True)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a mixed-effects linear model predicting log-transformed nut-cracking efficiency.

    Primary model (preferred): linear mixed-effects model with random intercept for chimpanzee
    to account for repeated measures per individual. Fixed effects: age, sex_male, help_yes,
    and hammer (categorical control).

    Returns the fitted model object (MixedLMResults) and prints a summary.
    Falls back to OLS with cluster-robust standard errors by chimpanzee if MixedLM fails to converge.
    """
    # Make a local copy
    df = df.copy()

    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    formula = 'LogEfficiency ~ age + sex_male + help_yes + C(hammer)'

    try:
        # Mixed effects model with random intercept for each chimpanzee
        md = smf.mixedlm(formula, data=df, groups=df['chimpanzee'])
        mdf = md.fit(reml=False)
        print('MixedLM fit successful. Summary:')
        print(mdf.summary())
        return mdf
    except Exception as e:
        # Fallback: OLS on the same fixed effects with cluster-robust SEs by chimpanzee
        print('MixedLM failed with error:', e)
        print('Falling back to OLS with cluster-robust SEs by chimpanzee.')

        # Build design matrix using patsy via formula API for consistency
        ols_model = smf.ols(formula, data=df).fit()
        # Get cluster-robust covariance (clusters = chimpanzee)
        robust = ols_model.get_robustcov_results(cov_type='cluster', groups=df['chimpanzee'])
        print(robust.summary())
        return robust


