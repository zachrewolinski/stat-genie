from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/panda_nuts/noperturb_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy to avoid modifying original
    df = df.copy()

    # Ensure required columns exist
    required_cols = ['chimpanzee', 'age', 'sex', 'nuts_opened', 'seconds', 'help']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input dataframe: {missing}")

    # Drop rows with missing key outcome or exposure
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age', 'sex', 'help'])

    # Remove sessions with non-positive seconds (cannot be used as exposure)
    df = df[df['seconds'] > 0].copy()

    # Create binary sex indicator: male = 1, female = 0
    # Accept common variants in case of capitalization
    def map_sex(s):
        if pd.isna(s):
            return np.nan
        s_str = str(s).strip().lower()
        if s_str in ['m', 'male']:
            return 1
        if s_str in ['f', 'female']:
            return 0
        # fallback: try to interpret single-letter
        return np.nan
    df['sex_m'] = df['sex'].apply(map_sex)

    # Create binary help indicator: yes = 1, no = 0
    def map_help(h):
        if pd.isna(h):
            return np.nan
        h_str = str(h).strip().lower()
        if h_str in ['y', 'yes']:
            return 1
        if h_str in ['n', 'no']:
            return 0
        return np.nan
    df['help_y'] = df['help'].apply(map_help)

    # Drop rows where mapping failed
    df = df.dropna(subset=['sex_m', 'help_y'])

    # Convert chimpanzee id to integer (if possible) and keep as-is for clustering
    # If IDs are non-numeric, keep them as categorical strings
    try:
        df['chimpanzee'] = df['chimpanzee'].astype(int)
    except Exception:
        df['chimpanzee'] = df['chimpanzee'].astype(str)

    # Center age for interpretability
    df['age_c'] = df['age'].astype(float) - df['age'].astype(float).mean()

    # Exposure: log(seconds) for use as offset in count models
    df['log_seconds'] = np.log(df['seconds'].astype(float))

    # Derived efficiency measures for exploratory analysis (not used by the main model but kept in output)
    df['EfficiencyRate'] = df['nuts_opened'].astype(float) / df['seconds'].astype(float)
    # log1p avoids -inf for zero rates
    df['log_efficiency'] = np.log1p(df['EfficiencyRate'])

    # Final columns to keep (but return full df to preserve any other metadata)
    # Ensure correct dtypes
    df['nuts_opened'] = df['nuts_opened'].astype(int)
    df['sex_m'] = df['sex_m'].astype(int)
    df['help_y'] = df['help_y'].astype(int)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit a negative binomial generalized linear model predicting the count of nuts opened
    with session duration as exposure (log_seconds as offset). Use clustered robust
    standard errors at the chimpanzee level to account for repeated measures.

    Model: nuts_opened ~ age_c + sex_m + help_y + offset(log_seconds)

    Returns a dictionary with the fitted model and a clustered-robust-results object.
    """
    # Prepare model matrices
    X = df[['age_c', 'sex_m', 'help_y']].copy()
    X = sm.add_constant(X)
    y = df['nuts_opened']
    offset = df['log_seconds']

    # Fit GLM Negative Binomial with offset
    # Note: statsmodels' NegativeBinomial family in GLM uses a parameterization that
    # may differ from other implementations; this is appropriate for overdispersed counts.
    glm_nb = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset)
    res_nb = glm_nb.fit()

    # Compute cluster-robust standard errors clustered by chimpanzee id
    # If chimpanzee column is numeric or string, this works with groups argument
    try:
        res_nb_cluster = res_nb.get_robustcov_results(cov_type='cluster', groups=df['chimpanzee'])
    except Exception as e:
        # If clustering fails, fall back to the original results and include the exception message
        res_nb_cluster = res_nb
        res_nb_cluster._cluster_failure = str(e)

    # Return both the plain fit and the clustered-robust-results (clustered is primary for inference)
    return {
        'glm_nb_result': res_nb,
        'glm_nb_clustered_result': res_nb_cluster
    }


