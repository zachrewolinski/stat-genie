from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/panda_nuts/add_features_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Required columns for analysis
    required_cols = ['nuts_opened', 'seconds', 'age', 'sex', 'help', 'hammer', 'chimpanzee']

    # Drop rows with missing core variables
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age', 'sex', 'help', 'hammer', 'chimpanzee'])

    # Ensure numeric types where expected
    df['nuts_opened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Drop any rows where conversion created NaNs
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age'])

    # Remove sessions with non-positive durations to avoid log(0) or negative exposure
    df = df[df['seconds'] > 0]

    # Create efficiency measure (nuts per second) for diagnostics
    df['efficiency'] = df['nuts_opened'] / df['seconds']

    # Create log(seconds) to use as offset in count model
    df['log_seconds'] = np.log(df['seconds'])

    # Standardize and encode sex -> sex_male (1 = male, 0 = female)
    df['sex'] = df['sex'].astype(str).str.strip().str.lower()
    df['sex_male'] = df['sex'].map({'m': 1, 'male': 1}).fillna(0).astype(int)

    # Encode help -> help_received (1 = received help, 0 = no help)
    df['help'] = df['help'].astype(str).str.strip().str.lower()
    df['help_received'] = df['help'].map({'y': 1, 'yes': 1, 'n': 0, 'no': 0}).fillna(0).astype(int)

    # Clean hammer as categorical string (keeps original values for later dummy-encoding in model)
    df['hammer'] = df['hammer'].astype(str).str.strip()

    # Ensure chimpanzee id is preserved (if it's numeric keep it; if string strip)
    # Leave column name as 'chimpanzee' for clustering
    if df['chimpanzee'].dtype == object:
        df['chimpanzee'] = df['chimpanzee'].astype(str).str.strip()

    # Final: drop any rows with infinite or NaN efficiency or log_seconds
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['efficiency', 'log_seconds'])

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # Build design matrix: include age (continuous), sex_male, help_received, and hammer (categorical -> dummies)
    # Hammer dummies are created here to ensure we control for tool type. We drop the first category to avoid multicollinearity.
    X_base = df[['age', 'sex_male', 'help_received', 'hammer']].copy()
    hammer_dummies = pd.get_dummies(X_base['hammer'], prefix='hammer', drop_first=True)

    X = pd.concat([X_base[['age', 'sex_male', 'help_received']].astype(float), hammer_dummies.astype(float)], axis=1)
    X = sm.add_constant(X)

    # Outcome and offset
    y = df['nuts_opened'].astype(float)
    offset = df['log_seconds']

    # Fit a Poisson GLM with log(seconds) as offset -> models rate (nuts per second)
    poisson_model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
    res = poisson_model.fit()

    # Obtain cluster-robust SEs clustered by individual chimpanzee ID (accounts for repeated measures)
    # If chimpanzee identifiers are strings, groups accepts them.
    try:
        res_robust = res.get_robustcov_results(cov_type='cluster', groups=df['chimpanzee'])
    except Exception:
        # Fall back to default (non-clustered) results if clustering fails
        res_robust = res

    # Return the fitted results with cluster-robust covariances (if computed)
    return res_robust


