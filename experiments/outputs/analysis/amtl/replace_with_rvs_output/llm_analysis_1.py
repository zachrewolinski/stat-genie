from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/replace_with_rvs_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset into the dataframe used for binomial GLM.

    Produces the following new columns used in modeling:
    - amtl_success: number of missing teeth for the observation (num_amtl)
    - amtl_failure: number of observed non-missing sockets (sockets - num_amtl)
    - genus_Homo, genus_Pongo, genus_Papio: genus dummy indicators (reference = Pan)
    - tooth_Anterior, tooth_Posterior: tooth class dummy indicators (reference = Premolar)
    - age_c: mean-centered age
    - prob_male: preserved from original data (proxy for sex)
    - specimen: preserved for clustering
    """
    df = df.copy()

    # Required columns
    required = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen']
    missing_req = [c for c in required if c not in df.columns]
    if missing_req:
        raise ValueError(f"Input dataframe missing required columns: {missing_req}")

    # Drop rows with missing essential values
    df = df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen'])

    # Ensure numeric columns are numeric
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')

    # Drop rows that became NA after coercion
    df = df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male'])

    # Remove impossible rows: sockets must be positive integer and >= num_amtl
    df = df[df['sockets'] >= 1]
    df = df[df['num_amtl'] >= 0]
    df = df[df['sockets'] >= df['num_amtl']]

    # Convert counts to integer type
    df['amtl_success'] = df['num_amtl'].astype(int)
    df['amtl_failure'] = (df['sockets'] - df['num_amtl']).astype(int)

    # Create genus dummies. Reference category: Pan (chimpanzees)
    df['genus_Homo'] = (df['genus'].astype(str) == 'Homo sapiens').astype(int)
    df['genus_Pongo'] = (df['genus'].astype(str) == 'Pongo').astype(int)
    df['genus_Papio'] = (df['genus'].astype(str) == 'Papio').astype(int)
    # Note: Pan (and any other genera not matched above) is implicitly the reference

    # Create tooth class dummies. Reference category: Premolar
    df['tooth_Anterior'] = (df['tooth_class'].astype(str) == 'Anterior').astype(int)
    df['tooth_Posterior'] = (df['tooth_class'].astype(str) == 'Posterior').astype(int)

    # Mean-center age
    df['age_c'] = df['age'] - df['age'].mean()

    # Ensure specimen column is string for clustering
    df['specimen'] = df['specimen'].astype(str)

    # Keep only columns needed for modeling plus specimen for clustering and some diagnostics
    keep_cols = [
        'amtl_success', 'amtl_failure',
        'genus_Homo', 'genus_Pongo', 'genus_Papio',
        'tooth_Anterior', 'tooth_Posterior',
        'age_c', 'prob_male', 'specimen',
        # keep original for reference/diagnostics
        'num_amtl', 'sockets', 'age', 'genus', 'tooth_class'
    ]
    # Some of these may not exist if original df omitted them; filter existing
    keep_cols = [c for c in keep_cols if c in df.columns]

    return df[keep_cols].reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a binomial GLM for AMTL counts with controls.

    Model formula (in design-matrix form):
    logit(p) = const + beta1*genus_Homo + beta2*genus_Pongo + beta3*genus_Papio
               + beta4*tooth_Anterior + beta5*tooth_Posterior + beta6*age_c + beta7*prob_male

    The response is provided as a two-column array [successes, failures] so that varying numbers
    of sockets (trials) are accounted for directly.

    Returns a dictionary with:
    - 'glm_result': the fitted GLM result (default covariance)
    - 'glm_result_clustered': result with cluster-robust SEs clustered by specimen (if possible)
    """
    # Build the binomial response as a (n,2) array: [successes, failures]
    endog = np.column_stack([df['amtl_success'].values, df['amtl_failure'].values])

    # Design matrix
    exog_cols = ['genus_Homo', 'genus_Pongo', 'genus_Papio', 'tooth_Anterior', 'tooth_Posterior', 'age_c', 'prob_male']
    missing_exog = [c for c in exog_cols if c not in df.columns]
    if missing_exog:
        raise ValueError(f"Missing required design columns: {missing_exog}")

    exog = df[exog_cols].astype(float)
    exog = sm.add_constant(exog, has_constant='add')

    # Fit GLM (Binomial family)
    glm = sm.GLM(endog, exog, family=sm.families.Binomial())
    res = glm.fit()

    # Attempt cluster-robust SEs by specimen (to account for within-specimen correlation)
    clustered_result = None
    if 'specimen' in df.columns:
        try:
            clustered_result = res.get_robustcov_results(cov_type='cluster', groups=df['specimen'].values)
        except Exception:
            # If clustering fails for any reason, keep None and return the plain result
            clustered_result = None

    results = {
        'glm_result': res,
        'glm_result_clustered': clustered_result
    }

    return results


