from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/panda_nuts/replace_with_rvs_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transformations performed:
      - Clean and standardize sex and help columns to binary indicators (sex_M, help_yes).
      - Remove rows with invalid or zero session durations.
      - Compute nuts_per_sec = nuts_opened / seconds (nuts per second).
      - Ensure hammer and chimpanzee are categorical (hammer as category, chimpanzee for grouping).
      - Also create a log-transformed efficiency (log_nuts_per_sec) for diagnostics/robustness checks.

    Returns the dataframe with added columns used in modeling: ['nuts_per_sec', 'log_nuts_per_sec', 'sex_M', 'help_yes', 'hammer', 'chimpanzee']
    """

    # Work on a copy to avoid modifying original
    df = df.copy()

    # Standardize column names if necessary (no-op if already correct)
    # Ensure numeric columns are numeric
    df['nuts_opened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Normalize sex to lowercase strings then map to binary: male=1, female=0
    df['sex'] = df['sex'].astype(str).str.strip().str.lower()
    df['sex_M'] = df['sex'].map({'m': 1, 'male': 1, 'f': 0, 'female': 0})

    # Normalize help to lowercase and map yes/no to binary
    df['help'] = df['help'].astype(str).str.strip().str.lower()
    df['help_yes'] = df['help'].map({'y': 1, 'yes': 1, 'n': 0, 'no': 0})

    # Drop rows with invalid or zero seconds or missing nuts_opened, seconds
    df = df[~df['seconds'].isna()]
    df = df[~df['nuts_opened'].isna()]
    # Remove non-positive durations
    df = df[df['seconds'] > 0]

    # Compute nuts per second (efficiency)
    df['nuts_per_sec'] = df['nuts_opened'] / df['seconds']

    # Log-transform (add small constant to avoid log(0)) for diagnostics/robustness
    df['log_nuts_per_sec'] = np.log1p(df['nuts_per_sec'])

    # Ensure hammer treated as categorical
    if 'hammer' in df.columns:
        df['hammer'] = df['hammer'].astype('category')
    else:
        # If hammer is missing in the schema, create a sentinel
        df['hammer'] = pd.Series(['unknown'] * len(df), dtype='category')

    # Ensure chimpanzee id is categorical / group identifier
    df['chimpanzee'] = df['chimpanzee'].astype('category')

    # It's fine to keep original raw columns (nuts_opened, seconds) for transparency

    # Final drop: drop rows with missing predictor variables used in the model
    df = df.dropna(subset=['age', 'sex_M', 'help_yes', 'nuts_per_sec'])

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a linear mixed-effects model predicting nut-cracking efficiency (nuts_per_sec) from
    age, sex, and help, controlling for hammer type and including a random intercept for chimpanzee.

    Primary model: mixed effects model (random intercept for chimpanzee)
      nuts_per_sec ~ age + sex_M + help_yes + C(hammer) + (1 | chimpanzee)

    If MixedLM fails for any reason, fall back to OLS with cluster-robust standard errors clustered by chimpanzee.

    Returns the fitted results object (MixedLMResults or RegressionResultsWrapper).
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Prepare data: drop rows missing variables required for modeling
    model_df = df.dropna(subset=['nuts_per_sec', 'age', 'sex_M', 'help_yes', 'hammer', 'chimpanzee'])

    # Ensure hammer is treated as categorical for the formula
    model_df['hammer'] = model_df['hammer'].astype('category')

    formula = 'nuts_per_sec ~ age + sex_M + help_yes + C(hammer)'

    try:
        # Fit mixed effects model with random intercept for chimpanzee
        md = smf.mixedlm(formula, data=model_df, groups=model_df['chimpanzee'])
        mdf = md.fit(reml=False)
        print('\nMixedLM fit successful. Summary:')
        print(mdf.summary())
        return mdf
    except Exception as e:
        # Fallback: OLS with cluster-robust SEs by chimpanzee
        print('\nMixedLM failed with error:', e)
        print('Falling back to OLS with cluster-robust standard errors (clustered by chimpanzee).')

        ols_mod = smf.ols(formula, data=model_df).fit()

        # Compute clustered robust covariance (by chimpanzee)
        try:
            clustered = ols_mod.get_robustcov_results(cov_type='cluster', groups=model_df['chimpanzee'])
            print('\nOLS clustered robust SE results:')
            print(clustered.summary())
            return clustered
        except Exception as e2:
            print('Clustered robust SE computation failed:', e2)
            print('\nReturning plain OLS results:')
            print(ols_mod.summary())
            return ols_mod


