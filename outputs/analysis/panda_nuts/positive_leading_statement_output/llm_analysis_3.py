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
    # Copy to avoid modifying original
    df = df.copy()

    # Drop rows with missing / invalid key session measures
    df = df.dropna(subset=['nuts_opened', 'seconds'])

    # Remove sessions with non-positive duration
    df = df[df['seconds'] > 0]

    # Standardize textual columns and create binary indicators
    # sex_male: 1 if sex == 'm' (case insensitive), else 0
    df['sex_male'] = df['sex'].astype(str).str.lower().apply(lambda x: 1 if x == 'm' else 0)

    # help_yes: 1 if help indicates yes ('y' or 'Y' or 'yes'), else 0
    df['help_clean'] = df['help'].astype(str).str.lower()
    df['help_yes'] = df['help_clean'].apply(lambda x: 1 if x in ['y', 'yes', 'true', '1'] else 0)
    df.drop(columns=['help_clean'], inplace=True)

    # Compute raw efficiency: nuts per minute
    df['Efficiency_npm'] = df['nuts_opened'] / df['seconds'] * 60.0

    # Log-transform efficiency to stabilize variance and downweight outliers
    # Use log1p to handle zero efficiency
    df['log_efficiency'] = np.log1p(df['Efficiency_npm'])

    # Center age (z_age)
    # If missing ages are present, they will remain NA; higher-level code can drop them or model will drop rows
    if 'age' in df.columns:
        df['z_age'] = df['age'] - df['age'].mean()
    else:
        # create a placeholder if age missing (all NaN)
        df['z_age'] = np.nan

    # Create explicit hammer-type dummy columns so column names are deterministic in the model
    # If particular hammer types are not present in the data, these columns will be all zeros.
    df['hammer_str'] = df['hammer'].astype(str).str.lower()
    df['hammer_Q'] = df['hammer_str'].apply(lambda x: 1 if x == 'q' else 0)
    df['hammer_G'] = df['hammer_str'].apply(lambda x: 1 if x == 'g' else 0)
    df['hammer_wood'] = df['hammer_str'].apply(lambda x: 1 if x == 'wood' else 0)
    df.drop(columns=['hammer_str'], inplace=True)

    # Ensure chimpanzee column exists and is suitable as group id (keep original IDs)
    # If chimpanzee IDs are not numeric, leave as-is (mixedlm accepts categorical grouping)
    df['chimpanzee'] = df['chimpanzee']

    # Final: drop rows with missing values in dependent or primary independent vars
    df = df.dropna(subset=['log_efficiency', 'z_age', 'sex_male', 'help_yes'])

    # Return dataframe containing all columns used in modeling
    # (we keep other columns for future checks if needed)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a linear mixed-effects model predicting log-transformed efficiency from age, sex, and help,
    controlling for hammer type and with a random intercept for each chimpanzee.

    Returns:
      - If MixedLM fits successfully: the fitted MixedLMResults object
      - If MixedLM fails: fallback OLS with cluster-robust standard errors by chimpanzee (RegressionResultsWrapper)
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Build formula (explicit hammer dummies are included)
    formula = 'log_efficiency ~ z_age + sex_male + help_yes + hammer_Q + hammer_G + hammer_wood'

    # Try mixed-effects model with random intercept for chimpanzee
    try:
        md = smf.mixedlm(formula, df, groups=df['chimpanzee'])
        mdf = md.fit(reml=False)  # use ML for likelihood-based comparison if needed
        # Return the fitted mixed model object; users can call mdf.summary()
        return mdf
    except Exception as e:
        # Fallback: OLS with cluster-robust (by chimpanzee) SEs
        # Prepare design matrix
        y = df['log_efficiency']
        X = sm.add_constant(df[['z_age', 'sex_male', 'help_yes', 'hammer_Q', 'hammer_G', 'hammer_wood']])
        ols_mod = sm.OLS(y, X).fit()
        try:
            ols_cl = ols_mod.get_robustcov_results(cov_type='cluster', groups=df['chimpanzee'])
            return ols_cl
        except Exception:
            # If clustering fails, return plain OLS
            return ols_mod


