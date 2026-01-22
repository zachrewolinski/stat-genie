from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/panda_nuts/replace_and_positive_statement_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset to produce the columns required for modeling.

    Output columns used in the model:
      - eff_nuts_per_min: nuts opened per minute (raw efficiency, kept for inspection)
      - log_eff_nuts_per_min: log-transformed efficiency (dependent variable)
      - age_c: mean-centered age (independent)
      - sex_male: binary indicator (1 = male, 0 = female) (independent)
      - help_yes: binary indicator for receiving help (1 = yes, 0 = no) (independent)
      - hammer: hammer type (categorical control)
      - chimpanzee: chimpanzee ID (grouping variable / control)
    """
    df = df.copy()

    # Drop rows missing core variables needed to compute efficiency and predictors
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age', 'sex', 'help', 'chimpanzee'])

    # Remove sessions with non-positive duration to avoid division by zero
    df = df[df['seconds'] > 0]

    # Standardize and create binary variables for 'help' and 'sex'
    # 'help' values in dataset appear as 'y' and 'N' (case inconsistent). Map 'y'/'Y' to 1, else 0.
    df['help_yes'] = df['help'].astype(str).str.strip().str.lower().map({'y': 1, 'yes': 1}).fillna(0).astype(int)

    # Sex: map 'm' or 'male' (case-insensitive) to 1, everything else -> 0 (female)
    df['sex_male'] = df['sex'].astype(str).str.strip().str.lower().map({'m': 1, 'male': 1}).fillna(0).astype(int)

    # Efficiency: nuts opened per minute
    df['eff_nuts_per_min'] = df['nuts_opened'] / df['seconds'] * 60.0

    # Address possible zeros / skew by log-transforming efficiency. Add tiny constant to avoid log(0).
    df['log_eff_nuts_per_min'] = np.log(df['eff_nuts_per_min'] + 1e-6)

    # Center age for interpretability
    df['age_c'] = df['age'] - df['age'].mean()

    # Ensure hammer is a string categorical column (Patsy/statsmodels will handle C(hammer))
    df['hammer'] = df['hammer'].astype(str)

    # Ensure chimpanzee ID is an integer (grouping variable)
    df['chimpanzee'] = df['chimpanzee'].astype(int)

    # Optional: keep only sensible values for efficiency (filter out extreme negative/NaN log values)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['log_eff_nuts_per_min'])

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a mixed-effects linear model predicting log-transformed nut-cracking efficiency.

    Primary predictors (fixed effects): age_c, sex_male, help_yes
    Controls (fixed effects): hammer (categorical)
    Grouping (random effect): random intercept for chimpanzee

    Returns the fitted model object (MixedLMResults if successful). If MixedLM fails,
    falls back to OLS with clustered standard errors by chimpanzee and returns the OLS result.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Formula: include hammer as categorical control. Interaction terms are not requested by the
    # research question; we're testing main effects of age, sex, and help.
    formula = 'log_eff_nuts_per_min ~ age_c + sex_male + help_yes + C(hammer)'

    # Try a mixed-effects model with a random intercept per chimpanzee to account for repeated measures
    try:
        md = smf.mixedlm(formula, df, groups=df['chimpanzee'])
        mdf = md.fit(reml=False)
        print('MixedLM fit successful')
        print(mdf.summary())
        return mdf
    except Exception as e:
        # Fallback: OLS with cluster-robust SEs by chimpanzee
        print('MixedLM failed with error:', str(e))
        print('Falling back to OLS with cluster-robust standard errors by chimpanzee')
        ols = smf.ols(formula, data=df).fit()
        # Compute clustered cov_type summary if desired
        try:
            clustered = ols.get_robustcov_results(cov_type='cluster', groups=df['chimpanzee'])
            print(clustered.summary())
            return clustered
        except Exception:
            print('Clustered SE computation failed; returning plain OLS fit')
            print(ols.summary())
            return ols


