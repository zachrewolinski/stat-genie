from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/panda_nuts/noperturb_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into analysis-ready dataframe.

    Creates the following columns used in modeling:
      - rate: nuts_opened / seconds
      - log_rate: natural log of rate (with small lower bound to avoid log(0))
      - age_c: centered age
      - sex_M: male indicator (1=male, 0=female/other)
      - help_yes: help indicator (1=yes, 0=no/other)
      - hammer: ensured as categorical (kept original values)
      - chimpanzee: ensured as string/categorical grouping variable

    Drops rows with missing or invalid values for the required columns.
    """
    df = df.copy()

    # Required columns for this analysis
    required_cols = ['nuts_opened', 'seconds', 'age', 'sex', 'help', 'hammer', 'chimpanzee']
    # Drop rows missing any required column
    df = df.dropna(subset=required_cols)

    # Remove sessions with non-positive session durations
    df = df[df['seconds'] > 0]

    # Compute raw rate (nuts per second)
    df['rate'] = df['nuts_opened'] / df['seconds']

    # Avoid zeros/negative rates before log-transform: clip to a tiny positive value
    eps = 1e-6
    df['rate'] = df['rate'].clip(lower=eps)

    # Log-transform the rate to stabilize variance
    df['log_rate'] = np.log(df['rate'])

    # Center age (helps interpret interaction and intercept)
    df['age_c'] = df['age'] - df['age'].mean()

    # Encode sex -> male indicator (robust to capitalization and slightly flexible values)
    df['sex_M'] = (
        df['sex'].astype(str).str.strip().str.lower()
        .map(lambda s: 1 if s in ('m', 'male') else 0)
        .astype(int)
    )

    # Encode help -> yes indicator
    df['help_yes'] = (
        df['help'].astype(str).str.strip().str.lower()
        .map(lambda s: 1 if s in ('y', 'yes', 'true') else 0)
        .astype(int)
    )

    # Keep hammer as categorical factor for modeling (ensure dtype is object/categorical)
    df['hammer'] = df['hammer'].astype('category')

    # Ensure chimpanzee grouping column is a string (consistent groups)
    df['chimpanzee'] = df['chimpanzee'].astype(str)

    # Return only columns that will be used in modeling plus a few helpful ones
    keep_cols = ['chimpanzee', 'age', 'age_c', 'sex', 'sex_M', 'help', 'help_yes', 'hammer', 'nuts_opened', 'seconds', 'rate', 'log_rate']
    # Some of these may not exist if original df was missing them; guard by intersecting
    keep_cols = [c for c in keep_cols if c in df.columns]

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a linear mixed-effects model predicting log_rate (log nuts/sec).

    Model specification:
      log_rate ~ age_c + sex_M + help_yes + age_c:help_yes + C(hammer) + (1 | chimpanzee)

    - Fixed effects: centered age, sex (male indicator), help (yes indicator), their interaction (age x help),
      and hammer type as a categorical control.
    - Random effects: random intercept for chimpanzee (to account for repeated measures / individual baselines).

    Returns the fitted statsmodels results object (MixedLMResults).
    """
    import statsmodels.formula.api as smf

    # Ensure the necessary transformed columns exist
    required = ['log_rate', 'age_c', 'sex_M', 'help_yes', 'hammer', 'chimpanzee']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns required for modeling: {missing}")

    # Formula with interaction between age and help and hammer as categorical control
    formula = 'log_rate ~ age_c + sex_M + help_yes + age_c:help_yes + C(hammer)'

    # Fit mixed linear model with random intercept for chimpanzee
    md = smf.mixedlm(formula, data=df, groups=df['chimpanzee'])
    results = md.fit(reml=False)

    # Return the fitted results object for further inspection (summary, coef, etc.)
    return results


