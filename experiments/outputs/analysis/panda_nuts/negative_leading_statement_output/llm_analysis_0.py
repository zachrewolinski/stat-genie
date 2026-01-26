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
    Transform the raw dataset into a modeling-ready dataframe. Creates standardized age, binary sex/help indicators,
    hammer-type dummy columns (explicit names), efficiency measure, and log(seconds) for Poisson offset.

    Final dataframe columns used by the models:
      - chimpanzee (as in input)
      - age, age_std
      - sex, sex_m
      - help, help_y
      - hammer (original), hammer_Q, hammer_G, hammer_wood
      - seconds, log_seconds
      - nuts_opened
      - efficiency
    """

    # Work on a copy
    df = df.copy()

    # Drop rows missing core variables needed for modeling
    required_cols = ['nuts_opened', 'seconds', 'age', 'sex', 'help', 'hammer', 'chimpanzee']
    df = df.dropna(subset=required_cols)

    # Ensure numeric types where expected
    df['nuts_opened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Remove rows with non-positive session durations (cannot be used as exposure)
    df = df[df['seconds'] > 0]

    # Create efficiency (nuts per second) for descriptive checks and mixed-model
    df['efficiency'] = df['nuts_opened'] / df['seconds']

    # Log seconds for offset in Poisson modeling (offset expects log(exposure))
    df['log_seconds'] = np.log(df['seconds'])

    # Binary help indicator: map common yes values to 1, else 0
    df['help_y'] = df['help'].astype(str).str.lower().isin(['y', 'yes']).astype(int)

    # Binary sex indicator: male = 1, female = 0. Be robust to case differences.
    df['sex_m'] = df['sex'].astype(str).str.lower().map({'m': 1, 'male': 1}).fillna(0).astype(int)

    # Standardize (z-score) age for stable coefficient interpretation
    df['age_std'] = (df['age'] - df['age'].mean()) / (df['age'].std(ddof=0) if df['age'].std(ddof=0) != 0 else 1)

    # Create explicit hammer-type dummies with deterministic column names. If a hammer type is absent in the data,
    # the column will be created and filled with zeros.
    for ham in ['Q', 'G', 'wood']:
        col = f'hammer_{ham}'
        df[col] = (df['hammer'].astype(str) == ham).astype(int)

    # Keep only relevant columns (but preserve originals for transparency)
    keep_cols = [
        'chimpanzee', 'age', 'age_std', 'sex', 'sex_m', 'help', 'help_y',
        'hammer', 'hammer_Q', 'hammer_G', 'hammer_wood',
        'nuts_opened', 'seconds', 'log_seconds', 'efficiency'
    ]
    # Some of these columns may not exist if original had unexpected categories; ensure existence
    existent_keep = [c for c in keep_cols if c in df.columns]
    df = df[existent_keep]

    # Reset index
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a primary Poisson regression modeling nuts_opened as a count with log(seconds) as an offset (exposure).
    Predictors: standardized age (age_std), sex (sex_m), help (help_y), and hammer dummies (hammer_Q, hammer_G, hammer_wood).
    Use cluster-robust standard errors clustered by chimpanzee to account for repeated observations per individual.

    Secondary check: fit a linear mixed-effects model predicting continuous efficiency (nuts/sec) with a random intercept
    for chimpanzee using the main predictors (age_std, sex_m, help_y). This provides a complementary viewpoint
    that relaxes the count-model assumptions.

    Returns a dictionary with fitted model objects and prints summaries.
    """

    results = {}

    # Ensure the columns we expect exist in the dataframe
    required_for_poisson = ['nuts_opened', 'seconds', 'log_seconds', 'age_std', 'sex_m', 'help_y']
    if not all(col in df.columns for col in required_for_poisson):
        missing = [col for col in required_for_poisson if col not in df.columns]
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Build design matrix for Poisson model. Use hammer dummies if present.
    predictor_cols = ['age_std', 'sex_m', 'help_y']
    for ham_col in ['hammer_Q', 'hammer_G', 'hammer_wood']:
        if ham_col in df.columns:
            predictor_cols.append(ham_col)

    X = df[predictor_cols]
    X = sm.add_constant(X, has_constant='add')
    y = df['nuts_opened']
    offset = df['log_seconds']

    # Fit Poisson GLM with offset
    poisson_model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
    poisson_res = poisson_model.fit()

    # Obtain cluster-robust standard errors clustered by chimpanzee id
    try:
        poisson_cluster_res = poisson_res.get_robustcov_results(cov_type='cluster', groups=df['chimpanzee'])
    except Exception:
        # If clustering fails for any reason, fall back to default results
        poisson_cluster_res = poisson_res

    print('\nPoisson GLM (nuts_opened with log(seconds) offset) - coefficients and cluster-robust SE (clustered by chimpanzee):\n')
    print(poisson_cluster_res.summary())

    results['poisson'] = poisson_cluster_res
    results['poisson_raw'] = poisson_res

    # Secondary check: linear mixed-effects model on continuous efficiency (nuts/sec)
    # Use a simple fixed-effects structure with a random intercept per chimpanzee.
    if 'efficiency' in df.columns:
        mixed_predictors = ['age_std', 'sex_m', 'help_y']
        mixed_X = df[mixed_predictors]
        mixed_X = sm.add_constant(mixed_X, has_constant='add')

        try:
            mixed_model = sm.MixedLM(df['efficiency'], mixed_X, groups=df['chimpanzee'])
            mixed_res = mixed_model.fit(reml=False)
            print('\nLinear mixed-effects model (efficiency = nuts/sec) with random intercept by chimpanzee:\n')
            print(mixed_res.summary())
            results['mixedlm'] = mixed_res
        except Exception as e:
            # If MixedLM fails (small sample issues), include the exception in results
            results['mixedlm_error'] = str(e)
    else:
        results['mixedlm_error'] = 'efficiency column not present'

    return results


