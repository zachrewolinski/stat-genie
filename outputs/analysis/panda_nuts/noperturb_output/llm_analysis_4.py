from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/panda_nuts/noperturb_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a modeling-ready dataframe.

    Produces the following key columns used in modeling:
      - nuts_per_sec: nuts_opened / seconds (float)
      - log_rate: log(nuts_opened + 0.5) - log(seconds) (float) -- dependent variable
      - age_c: centered age (age - mean(age)) (float)
      - sex_m: sex encoded as 1=male, 0=female (int)
      - help_y: help encoded as 1=yes, 0=no (int)
      - hammer_* : one-hot columns for hammer type (drop_first=True)
      - chimpanzee: string id used as grouping variable for mixed model
    """
    df = df.copy()

    # Ensure numeric columns are numeric
    df['nuts_opened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Drop rows missing core variables needed for the model
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age', 'sex', 'help'])

    # Normalize/encode sex: male -> 1, female -> 0 (case-insensitive)
    df['sex_str'] = df['sex'].astype(str).str.lower().str.strip()
    df['sex_m'] = df['sex_str'].map({'m': 1, 'male': 1, 'f': 0, 'female': 0})

    # Encode help: yes -> 1, no -> 0 (case-insensitive). Accept common variants.
    df['help_str'] = df['help'].astype(str).str.lower().str.strip()
    df['help_y'] = df['help_str'].map({'y': 1, 'yes': 1, 'n': 0, 'no': 0})

    # If mapping produced NaNs (unexpected labels), coerce to 0 for 'no' if clearly not 'y'
    # but keep rows with explicit unknowns dropped below
    # Compute raw rate and log-rate dependent variable
    # Use a small pseudocount (0.5) to handle zero nuts_opened safely in log domain
    df['nuts_per_sec'] = df['nuts_opened'] / df['seconds']
    df['log_rate'] = np.log(df['nuts_opened'] + 0.5) - np.log(df['seconds'])

    # Center age
    df['age_c'] = df['age'] - df['age'].mean()

    # Create hammer dummies (drop_first to avoid perfect multicollinearity)
    # Fill missing hammer with explicit category 'missing'
    df['hammer'] = df['hammer'].astype(str).fillna('missing')
    hammer_dummies = pd.get_dummies(df['hammer'], prefix='hammer', drop_first=True)
    df = pd.concat([df, hammer_dummies], axis=1)

    # Ensure chimpanzee is a string/categorical group label for MixedLM
    df['chimpanzee'] = df['chimpanzee'].astype(str)

    # Final drop: remove rows with any NaN or infinite values in modelling columns
    model_cols = ['log_rate', 'age_c', 'sex_m', 'help_y', 'nuts_per_sec', 'chimpanzee'] + list(hammer_dummies.columns)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=model_cols)

    # Convert sex_m and help_y to integer dtype
    df['sex_m'] = df['sex_m'].astype(int)
    df['help_y'] = df['help_y'].astype(int)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a mixed-effects linear model predicting log_rate (log nuts-per-second).

    Model specification:
      log_rate ~ age_c + sex_m + help_y + (hammer dummies) + (1 | chimpanzee)

    We use a random intercept per chimpanzee to account for repeated measures.
    """
    # collect hammer dummy columns created in transform
    hammer_cols = [c for c in df.columns if c.startswith('hammer_')]

    exog_vars = ['age_c', 'sex_m', 'help_y'] + hammer_cols
    formula = 'log_rate ~ ' + ' + '.join(exog_vars)

    # Fit mixed linear model with random intercept per chimpanzee
    # Use reml=False for likelihood-based comparisons (common default for reporting)
    md = sm.MixedLM.from_formula(formula, groups='chimpanzee', data=df)
    mdf = md.fit(reml=False)

    # Print and return the fitted model object (statsmodels result)
    print(mdf.summary())
    return mdf


