from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/panda_nuts/add_features_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Standardize string columns: strip whitespace and lowercase where applicable
    if 'sex' in df.columns:
        df['sex'] = df['sex'].astype(str).str.strip().str.lower()
    if 'help' in df.columns:
        df['help'] = df['help'].astype(str).str.strip().str.lower()
    if 'hammer' in df.columns:
        df['hammer'] = df['hammer'].astype(str).str.strip()

    # Drop rows with missing core variables required for the analysis
    required = ['nuts_opened', 'seconds', 'age', 'sex', 'help', 'chimpanzee']
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError(f"Input dataframe is missing required columns: {missing_required}")

    df = df.dropna(subset=['nuts_opened', 'seconds', 'age', 'sex', 'help', 'chimpanzee'])

    # Ensure seconds > 0 to avoid division by zero
    df = df[df['seconds'] > 0].copy()

    # Compute raw efficiency: nuts opened per second
    df['efficiency'] = df['nuts_opened'] / df['seconds']

    # Small constant to avoid log(0); log-transform to stabilize distribution
    df['log_efficiency'] = np.log(df['efficiency'] + 1e-6)

    # Convert help to binary: 'y' or 'yes' -> 1, else 0
    df['help_binary'] = df['help'].str.lower().isin(['y', 'yes', 'true', '1']).astype(int)

    # Clean chimpanzee id: keep as-is but ensure categorical/grouping type
    df['chimpanzee'] = df['chimpanzee'].astype(str)

    # Ensure hammer is categorical; fill missing with explicit category 'unknown'
    df['hammer'] = df['hammer'].fillna('unknown').astype(str)

    # Ensure sex has expected values ('f'/'m'); keep as category
    df['sex'] = df['sex'].astype('category')

    # Final dataframe returned contains all columns needed for modeling
    # (log_efficiency, efficiency, age, sex, help_binary, hammer, chimpanzee)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a mixed-effects linear model predicting log_efficiency from age, sex, and help,
    controlling for hammer type as a fixed effect and including a random intercept per chimpanzee.

    Returns the fitted model results object. If MixedLM fails to converge, falls back to OLS
    using cluster-robust standard errors clustered by chimpanzee.
    """
    # Ensure transformed dataframe contains required columns
    required = ['log_efficiency', 'age', 'sex', 'help_binary', 'hammer', 'chimpanzee']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Transformed dataframe is missing required columns for modeling: {missing}")

    # Import formula API
    import statsmodels.formula.api as smf

    # Build formula: main effects of age, sex, help; control for hammer type
    formula = 'log_efficiency ~ age + C(sex) + help_binary + C(hammer)'

    # Try mixed-effects model with random intercept per chimpanzee
    try:
        md = smf.mixedlm(formula, df, groups=df['chimpanzee'])
        mdf = md.fit(reml=False)
        return mdf
    except Exception as e:
        # If MixedLM fails (convergence or other issues), fall back to OLS with cluster-robust SEs
        ols = smf.ols(formula, data=df).fit()
        # Compute cluster-robust standard errors clustered by chimpanzee
        try:
            clustered = ols.get_robustcov_results(cov_type='cluster', groups=df['chimpanzee'])
            return clustered
        except Exception:
            # If cluster-robust also fails, return the plain OLS fit
            return ols


