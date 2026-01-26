from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/panda_nuts/negative_leading_statement_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw nut-cracking dataset to produce variables for modeling.

    Produces:
    - efficiency: nuts_opened / seconds (nuts per second)
    - log_efficiency: log(efficiency + eps)
    - sex_m: 1 if sex == 'm', 0 if sex == 'f'
    - help_yes: 1 if help indicates yes, 0 if no

    Keeps original columns needed for interpretation: chimpanzee, age, sex, hammer, nuts_opened, seconds
    """
    df = df.copy()

    # Drop rows with missing essential fields
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age', 'sex', 'help', 'hammer', 'chimpanzee'])

    # Ensure seconds are positive and sensible; drop non-positive durations
    df = df[df['seconds'] > 0]

    # Compute efficiency: nuts opened per second
    df['efficiency'] = df['nuts_opened'] / df['seconds']

    # Small offset to avoid log(0); using a tiny epsilon because efficiency can legitimately be zero
    eps = 1e-6
    df['log_efficiency'] = np.log(df['efficiency'] + eps)

    # Binary encoding for sex: male = 1, female = 0. Handle possible capitalization and unexpected values.
    df['sex'] = df['sex'].astype(str).str.lower()
    df['sex_m'] = df['sex'].map({'m': 1, 'f': 0})
    # If any sex values are not m/f, mark as NaN and drop
    df = df.dropna(subset=['sex_m'])
    df['sex_m'] = df['sex_m'].astype(int)

    # Binary encoding for help: map common variants (case-insensitive) to 1/0
    df['help'] = df['help'].astype(str).str.lower()
    df['help_yes'] = df['help'].map({'y': 1, 'yes': 1, 'n': 0, 'no': 0})
    # If help values are ambiguous, drop those rows
    df = df.dropna(subset=['help_yes'])
    df['help_yes'] = df['help_yes'].astype(int)

    # Ensure chimpanzee ID is present and of integer type for grouping
    df['chimpanzee'] = df['chimpanzee'].astype(int)

    # Keep only columns necessary for modeling and interpretation
    keep_cols = [
        'chimpanzee', 'age', 'sex', 'sex_m', 'hammer', 'nuts_opened', 'seconds',
        'help', 'help_yes', 'efficiency', 'log_efficiency'
    ]
    df = df.loc[:, keep_cols]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit models to test whether age, sex, and receiving help predict nut-cracking efficiency.

    Primary model: linear mixed effects model predicting log_efficiency with fixed effects
    age, sex_m, help_yes, and hammer (categorical) and a random intercept for chimpanzee.

    Secondary model: OLS with cluster-robust standard errors clustered by chimpanzee for comparison.

    Returns a dict with fitted model objects and printed summaries.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    results = {}

    # Fit a linear mixed effects model (random intercept for chimpanzee)
    # Use categorical hammer via C(hammer) in the formula so dummies are constructed automatically
    formula = 'log_efficiency ~ age + sex_m + help_yes + C(hammer)'
    mixed = smf.mixedlm(formula, data=df, groups=df['chimpanzee'])
    mixed_fit = mixed.fit(reml=False)  # use ML for comparability
    results['mixedlm'] = mixed_fit

    # Also fit an OLS model for ease of interpretation and compute cluster-robust SEs by chimpanzee
    exog = sm.add_constant(df[['age', 'sex_m', 'help_yes']])
    ols = sm.OLS(df['log_efficiency'], exog).fit()
    # Cluster-robust covariance by chimpanzee
    try:
        ols_cluster = ols.get_robustcov_results(cov_type='cluster', groups=df['chimpanzee'])
    except Exception:
        # fallback: return standard OLS if cluster robust fails
        ols_cluster = ols
    results['ols_cluster'] = ols_cluster

    # Print model summaries for quick inspection (when this code is executed interactively)
    print('MixedLM summary:')
    print(mixed_fit.summary())
    print('\nOLS with cluster-robust SEs summary:')
    print(ols_cluster.summary())

    return results


