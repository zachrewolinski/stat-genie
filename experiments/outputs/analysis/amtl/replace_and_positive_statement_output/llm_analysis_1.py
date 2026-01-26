from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/replace_and_positive_statement_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw Gilmore (2013) AMTL dataset into a dataframe suitable for binomial regression.

    Produces the following columns required by the model:
    - amtl_successes: number of missing teeth for the given tooth_class (num_amtl)
    - amtl_failures: number of present/observable (not-missing) sockets = sockets - num_amtl
    - amtl_prop: proportion missing (amtl_successes / sockets)
    - is_human: 1 if genus == 'Homo sapiens', 0 otherwise
    - age_z: standardized age (z-score)
    - prob_male: preserved from input (used as continuous sex control)
    - tooth_class: preserved categorical variable (Anterio/Posterior/Premolar)
    - specimen: preserved identifier (for clustering)

    Also drops rows with invalid or missing values required for the model and removes rows where sockets < num_amtl.
    """
    # Make a shallow copy
    df = df.copy()

    # Required columns
    required_cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen']
    missing_req = [c for c in required_cols if c not in df.columns]
    if len(missing_req) > 0:
        raise ValueError(f"Input dataframe is missing required columns: {missing_req}")

    # Drop rows with NA in required columns
    df = df.dropna(subset=required_cols)

    # Ensure numeric types
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')

    # Drop rows that became NA after coercion
    df = df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male'])

    # Remove rows with impossible socket/count values
    df = df[df['sockets'] >= 1]
    df = df[df['num_amtl'] >= 0]
    df = df[df['num_amtl'] <= df['sockets']]

    # Create binomial outcome columns: successes and failures
    df['amtl_successes'] = df['num_amtl'].astype(int)
    df['amtl_failures'] = (df['sockets'] - df['num_amtl']).astype(int)
    df['amtl_prop'] = df['amtl_successes'] / df['sockets']

    # Create human indicator: 1 for modern Homo sapiens, 0 otherwise
    # Be robust to slightly different genus labels (e.g., 'Homo sapiens' vs 'Homo')
    df['genus'] = df['genus'].astype(str)
    df['is_human'] = df['genus'].apply(lambda x: 1 if x.strip().lower() in ['homo sapiens', 'homo'] else 0).astype(int)

    # Standardize age (z-score) for numerical stability
    df['age_z'] = (df['age'] - df['age'].mean()) / (df['age'].std(ddof=0) if df['age'].std(ddof=0) != 0 else 1.0)

    # Keep tooth_class as categorical but ensure consistent capitalization/spacing
    df['tooth_class'] = df['tooth_class'].astype(str).str.strip().str.title()

    # Ensure specimen is treated as string/identifier
    df['specimen'] = df['specimen'].astype(str)

    # Final selection: keep only columns needed for modeling + useful diagnostics
    keep_cols = ['specimen', 'is_human', 'amtl_successes', 'amtl_failures', 'amtl_prop', 'sockets', 'age', 'age_z', 'prob_male', 'tooth_class', 'genus', 'pop']
    existing_keep = [c for c in keep_cols if c in df.columns]
    df = df[existing_keep]

    # Reset index
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a binomial (logistic) regression for AMTL using successes/failures per row.

    Model specification (fixed-effects):
      amtl_successes / (amtl_successes + amtl_failures) ~ is_human + age_z + prob_male + C(tooth_class)

    Estimation details:
    - Uses statsmodels.GLM with Binomial family and endog passed as (successes, failures) pairs.
    - Computes cluster-robust standard errors clustered by 'specimen' to account for non-independence of rows from the same specimen.

    Returns the robust results object (statsmodels.results.Results class) for inspection.
    """
    # Basic checks
    required_model_cols = ['amtl_successes', 'amtl_failures', 'is_human', 'age_z', 'prob_male', 'tooth_class', 'specimen']
    missing = [c for c in required_model_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Transformed dataframe is missing required columns for modeling: {missing}")

    # Build endog as Nx2 array: [successes, failures]
    endog = np.vstack([df['amtl_successes'].astype(int), df['amtl_failures'].astype(int)]).T

    # Build exog (design matrix)
    # Start with core controls
    exog_df = df[['is_human', 'age_z', 'prob_male']].copy()

    # Add tooth_class dummies (drop first to avoid collinearity)
    tooth_dummies = pd.get_dummies(df['tooth_class'], prefix='tooth', drop_first=True)
    exog_df = pd.concat([exog_df, tooth_dummies], axis=1)

    # Add intercept
    exog = sm.add_constant(exog_df, has_constant='add')

    # Fit GLM binomial
    model_glm = sm.GLM(endog, exog, family=sm.families.Binomial())
    try:
        res = model_glm.fit()
    except Exception as e:
        # If fitting fails (e.g., perfect separation), raise informative error
        raise RuntimeError(f"GLM fit failed: {e}")

    # Compute cluster-robust standard errors clustered on specimen
    # Use get_robustcov_results to create a results object with clustered SEs
    try:
        res_cluster = res.get_robustcov_results(cov_type='cluster', groups=df['specimen'])
    except Exception as e:
        # If clustering fails, fall back to the original fit
        res_cluster = res

    # Print a concise summary (caller can inspect returned object for full details)
    print(res_cluster.summary())

    # Return the clustered results object
    return res_cluster


