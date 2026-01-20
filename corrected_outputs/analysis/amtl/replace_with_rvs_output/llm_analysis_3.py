from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import patsy


df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/amtl/replace_with_rvs_output/amtl.csv')


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataset for binomial regression of AMTL (num_amtl / sockets).

    Transformations performed:
    - Copy dataframe to avoid side-effects.
    - Ensure required columns are numeric / categorical and drop rows with missing critical values.
    - Remove rows with sockets <= 0 (cannot model trials = 0).
    - Cap num_amtl at sockets (safety) and convert to integer counts.
    - Create binary indicator 'is_human' (1 if genus contains 'Homo', 0 otherwise).
    - Make 'tooth_class' categorical.
    - Center age to create 'age_c'.
    - Create 'prop_amtl' as convenience (num_amtl / sockets).

    Returned dataframe contains all columns used in the model:
    'num_amtl', 'sockets', 'is_human', 'age_c', 'prob_male', 'tooth_class', 'specimen', plus 'prop_amtl'.
    """
    # Work on a copy
    df = df.copy()

    # Required columns
    required_cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen']

    # Coerce types where appropriate
    for c in ['num_amtl', 'sockets', 'age', 'prob_male']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing any required information for the model
    df = df.dropna(subset=required_cols)

    # Remove impossible/invalid rows: sockets must be positive
    df = df[df['sockets'] > 0]

    # Ensure integer counts and that num_amtl <= sockets
    # Convert sockets to int after removing invalid rows
    df['sockets'] = df['sockets'].astype(int)

    # Floor num_amtl (after filling NA with 0), then ensure integer and cap at sockets
    df['num_amtl'] = np.floor(df['num_amtl'].fillna(0)).astype(int)
    df['num_amtl'] = df[['num_amtl', 'sockets']].min(axis=1).astype(int)
    # Guard against negative values
    df.loc[df['num_amtl'] < 0, 'num_amtl'] = 0

    # Binary indicator: modern human vs non-human primate
    df['is_human'] = df['genus'].astype(str).str.contains('Homo', case=False, na=False).astype(int)

    # Tooth class as categorical
    df['tooth_class'] = df['tooth_class'].astype('category')

    # Center age to aid interpretation
    df['age_c'] = df['age'] - df['age'].mean()

    # Convenience column: observed proportion missing (for diagnostics/plots)
    df['prop_amtl'] = df['num_amtl'] / df['sockets']

    # Ensure required final columns exist (as specified by the contract)
    final_required = ['num_amtl', 'sockets', 'is_human', 'age_c', 'prob_male', 'tooth_class', 'specimen']
    missing = [c for c in final_required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after transform: {missing}")

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial GLM to test whether modern humans have different AMTL frequency
    compared to non-human primates, controlling for age, sex-probability, and tooth class.

    Approach:
    - Use statsmodels.GLM with Binomial family and provide endog as (successes, failures)
      so the binomial counts are correctly interpreted.
    - Predictors: is_human (0/1), age_c (centered age), prob_male, and C(tooth_class).
    - Compute cluster-robust standard errors clustered by specimen to account for non-independence.
    """
    # Work on a copy to avoid side effects
    df = df.copy()

    # Build endogenous as two-column array: [ successes, failures ]
    successes = df['num_amtl'].astype(int)
    failures = (df['sockets'] - df['num_amtl']).astype(int)
    # Ensure non-negative failures
    failures = failures.clip(lower=0)
    endog = np.vstack((successes, failures)).T

    # Build design matrix for predictors using patsy, include intercept by default
    # Keep only the conceptual predictor columns in the formula
    formula_rhs = 'is_human + age_c + prob_male + C(tooth_class)'
    exog = patsy.dmatrix(formula_rhs, df, return_type='dataframe')

    # Fit GLM with Binomial family using count-form endog
    glm_binom = sm.GLM(endog, exog, family=sm.families.Binomial()).fit()

    # Compute cluster-robust covariance (cluster by specimen)
    try:
        clustered = glm_binom.get_robustcov_results(cov_type='cluster', groups=df['specimen'])
    except Exception:
        clustered = None

    # Print brief summaries for user inspection
    print('\nStandard GLM (binomial) summary:')
    print(glm_binom.summary())

    if clustered is not None:
        print('\nCluster-robust results (clustered by specimen):')
        print(clustered.summary())
    else:
        print('\nCluster-robust covariance could not be computed; returning standard fit only.')

    return {
        'model': glm_binom,
        'clustered': clustered
    }