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
    Transform the raw dataset into a dataframe ready for modeling.

    Produces the following new/cleaned columns used in the model:
    - Efficiency: nuts_opened / seconds
    - LogEfficiency: log1p(Efficiency)
    - sex_male: binary indicator (1 if sex == 'm', 0 if sex == 'f')
    - help_yes: binary indicator (1 if help == 'y'/'yes', 0 otherwise)
    - hammer: categorical (kept as category)
    - chimpanzee: string category for grouping

    Rows with missing critical values or non-positive session durations are dropped.
    """
    df = df.copy()

    # Drop rows missing essential columns
    required = ['nuts_opened', 'seconds', 'age', 'sex', 'help', 'chimpanzee']
    df = df.dropna(subset=required)

    # Remove sessions with non-positive duration (cannot compute rate)
    df = df[df['seconds'] > 0].copy()

    # Compute efficiency (nuts opened per second) and log-transform
    df['Efficiency'] = df['nuts_opened'] / df['seconds']
    # Use log1p to handle zeros robustly
    df['LogEfficiency'] = np.log1p(df['Efficiency'])

    # Encode sex: male = 1, female = 0. If unknown values appear, map to NaN then fill with 0.
    df['sex_male'] = df['sex'].astype(str).str.lower().map({'m': 1, 'f': 0})
    df['sex_male'] = df['sex_male'].fillna(0).astype(int)

    # Encode help: 'y' or 'yes' -> 1, otherwise 0. Normalize strings first.
    df['help_yes'] = df['help'].astype(str).str.lower().isin(['y', 'yes']).astype(int)

    # Hammer as categorical control (keep original values but ensure dtype)
    df['hammer'] = df['hammer'].astype('category')

    # Ensure chimpanzee is a string/categorical group identifier
    df['chimpanzee'] = df['chimpanzee'].astype(str)

    # (Optional) Keep only columns needed for modeling plus relevant originals
    keep_cols = [
        'chimpanzee', 'age', 'sex', 'sex_male', 'help', 'help_yes', 'hammer',
        'nuts_opened', 'seconds', 'Efficiency', 'LogEfficiency'
    ]
    # Some datasets may not contain all original columns in keep_cols; intersect
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a mixed-effects model to test whether age, sex, and receiving help predict nut-cracking efficiency.

    Model specification (primary):
      LogEfficiency ~ age + sex_male + help_yes + C(hammer)
    with a random intercept for chimpanzee to account for repeated measures.

    Returns a dictionary with the fitted mixed-effects model and an OLS robustness fit.
    """
    import statsmodels.formula.api as smf

    df = df.copy()

    # Ensure the transformed columns exist
    required = ['LogEfficiency', 'age', 'sex_male', 'help_yes', 'hammer', 'chimpanzee']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Primary model: linear mixed effects with random intercept for chimpanzee
    formula = 'LogEfficiency ~ age + sex_male + help_yes + C(hammer)'
    md = smf.mixedlm(formula, df, groups=df['chimpanzee'])
    try:
        mdf = md.fit(reml=False)
    except Exception:
        # If convergence issues occur, try a simpler optimizer
        mdf = md.fit(reml=False, method='nm', maxiter=2000)

    # Robustness: ordinary least squares (ignores within-individual correlation)
    ols = smf.ols(formula, data=df).fit()

    # Return fitted model objects (users can print .summary() or inspect params)
    return {
        'mixedlm_result': mdf,
        'ols_result': ols
    }


