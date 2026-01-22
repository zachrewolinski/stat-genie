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
    Transform the original dataframe for modeling nut-cracking efficiency.

    Produces the following new/clean columns used in the model:
      - age: (ensures numeric)
      - sex_male: binary 1 = male, 0 = female
      - help_binary: binary 1 = received help, 0 = not
      - nuts_opened: ensured numeric (count)
      - seconds: ensured numeric (positive exposure)
      - nuts_per_sec: descriptive rate = nuts_opened / seconds
      - log_seconds: natural log of seconds (used as offset in count model)

    Drops rows with missing or invalid essential values.
    """
    df = df.copy()

    # Ensure essential columns exist
    required_cols = ['nuts_opened', 'seconds', 'age', 'sex', 'help']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Required column missing from input dataframe: {c}")

    # Coerce numeric columns and drop NA
    df['nuts_opened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    df = df.dropna(subset=['nuts_opened', 'seconds', 'age', 'sex', 'help'])

    # Remove impossible/degenerate sessions (non-positive time)
    df = df[df['seconds'] > 0]

    # Create binary indicator for help: treat 'y'/'Y'/'yes' as yes, everything else as no
    df['help_binary'] = (
        df['help'].astype(str)
        .str.strip()
        .str.lower()
        .map(lambda x: 1 if x in ['y', 'yes', 'true', 't', '1'] else 0)
    )

    # Create binary indicator for male sex
    df['sex_male'] = (
        df['sex'].astype(str)
        .str.strip()
        .str.lower()
        .map(lambda x: 1 if x in ['m', 'male'] else 0)
    )

    # Derived rate for descriptive purposes
    df['nuts_per_sec'] = df['nuts_opened'] / df['seconds']

    # Log seconds for use as an offset in count models
    # offset must be finite -> seconds > 0 ensured above
    df['log_seconds'] = np.log(df['seconds'])

    # Keep only columns we will use (plus a few for diagnostics)
    keep_cols = ['chimpanzee', 'age', 'sex', 'sex_male', 'help', 'help_binary',
                 'nuts_opened', 'seconds', 'log_seconds', 'nuts_per_sec', 'hammer']
    present_keep = [c for c in keep_cols if c in df.columns]
    df = df[present_keep]

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a count model for nuts_opened with session duration as exposure.

    Primary specification: Negative Binomial GLM for counts with log(seconds) as an offset.
    Predictors: age (continuous), sex_male (0/1), help_binary (0/1).

    Returns a dictionary containing the fitted model object and derived summaries (rate ratios).
    """
    df = df.copy()

    # Ensure transformed columns are present
    for c in ['nuts_opened', 'log_seconds', 'age', 'sex_male', 'help_binary']:
        if c not in df.columns:
            raise ValueError(f"Required column for modeling missing: {c}")

    # Prepare design matrix
    X = df[['age', 'sex_male', 'help_binary']].astype(float)
    X = sm.add_constant(X)
    y = df['nuts_opened'].astype(float)
    offset = df['log_seconds'].astype(float)

    # Fit Negative Binomial GLM with offset (models counts with exposure = seconds)
    # If the data are not overdispersed, this will still fit; NB is robust to overdispersion.
    nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset).fit()

    # Exponentiate coefficients to get multiplicative effects on the rate (rate ratios)
    params = nb_model.params
    conf_int = nb_model.conf_int()
    rate_ratios = np.exp(params)
    rr_conf_int = np.exp(conf_int)

    # Prepare a summary table (DataFrame) with coefficient, RR and CI
    summary_df = pd.DataFrame({
        'coef': params,
        'rr': rate_ratios,
        'ci_lower': rr_conf_int[0],
        'ci_upper': rr_conf_int[1],
        'pvalue': nb_model.pvalues
    })

    results = {
        'model_object': nb_model,
        'summary_table': summary_df,
        'aic': nb_model.aic,
        'deviance': nb_model.deviance
    }

    return results


