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
    # Make a copy to avoid modifying caller's df
    df = df.copy()

    # Required original columns: 'nuts_opened', 'seconds', 'age', 'sex', 'help', 'hammer', 'chimpanzee'
    # Drop rows with missing values in the core variables
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age', 'sex', 'help', 'hammer', 'chimpanzee'])

    # Ensure numeric columns are numeric
    df['nuts_opened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Drop rows where conversion failed
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age'])

    # Clean categorical text columns (strip whitespace, lower-case)
    df['sex'] = df['sex'].astype(str).str.strip().str.lower()
    df['help'] = df['help'].astype(str).str.strip().str.lower()
    # Keep hammer as-is (categorical) but ensure string type
    df['hammer'] = df['hammer'].astype(str).str.strip()

    # Create binary indicator for male sex: 1 if 'm', 0 if 'f' (if other values present they will be set to NaN and dropped)
    df.loc[~df['sex'].isin(['m', 'f']), 'sex'] = np.nan
    df['sex_male'] = (df['sex'] == 'm').astype(int)

    # Create binary indicator for help: 1 if 'y' (or 'yes'), 0 otherwise
    # Accept common variants 'y', 'yes'
    df['help_yes'] = df['help'].isin(['y', 'yes']).astype(int)

    # Compute nuts per minute as the efficiency measure
    # seconds should be > 0; drop unrealistic nonpositive durations
    df = df[df['seconds'] > 0]
    df['nuts_per_min'] = df['nuts_opened'] * 60.0 / df['seconds']

    # Use log1p transform to reduce skew and handle zeros
    df['log_nuts_per_min'] = np.log1p(df['nuts_per_min'])

    # Center age for interpretability in presence of interactions
    df['age_c'] = df['age'] - df['age'].mean()

    # Ensure chimpanzee ID is a categorical/grouping variable for mixed model
    # Keep original values but convert to string category (statsmodels accepts either)
    df['chimpanzee'] = df['chimpanzee'].astype(str)

    # Final selection: keep only rows with no missing values in derived columns
    df = df.dropna(subset=['log_nuts_per_min', 'age_c', 'sex_male', 'help_yes', 'hammer', 'chimpanzee'])

    # Return df with the new columns needed for modeling
    # (Also keep original columns for possible diagnostics)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """Run a mixed-effects model predicting log-transformed nuts-per-minute.

    Model specification:
      log_nuts_per_min ~ age_c + sex_male + help_yes + age_c:help_yes + sex_male:help_yes + C(hammer)
    Random effects: random intercept for chimpanzee (groups=df['chimpanzee']).

    Returns the fitted MixedLMResults object and prints a summary.
    """
    import statsmodels.formula.api as smf

    # Ensure required columns are present
    required = ['log_nuts_per_min', 'age_c', 'sex_male', 'help_yes', 'hammer', 'chimpanzee']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Define formula with interactions between help and the two focal predictors
    formula = 'log_nuts_per_min ~ age_c + sex_male + help_yes + age_c:help_yes + sex_male:help_yes + C(hammer)'

    # Fit a mixed-effects model with random intercept for each chimpanzee
    # Use REML=False to make results comparable to ML-based tests (common choice for comparisons)
    try:
        md = smf.mixedlm(formula, df, groups=df['chimpanzee'])
        mdf = md.fit(reml=False)
    except Exception as e:
        # If mixed model fails to converge, fall back to ordinary least squares with cluster-robust SEs
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        ols = smf.ols(formula, data=df).fit()
        # compute cluster robust SEs by chimpanzee
        clustered = ols.get_robustcov_results(cov_type='cluster', groups=df['chimpanzee'])
        print('MixedLM failed with error:', e)
        print('Falling back to OLS with cluster-robust SEs. Summary:')
        print(clustered.summary())
        return clustered

    # Print and return the fitted mixed model
    print(mdf.summary())
    return mdf


