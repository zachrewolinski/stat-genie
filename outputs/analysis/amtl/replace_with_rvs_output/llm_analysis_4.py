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
    Transform raw dataset to cleaned dataframe with the columns used in the model.

    Produces:
    - amtl_prop: num_amtl / sockets (proportion missing in the class)
    - num_amtl, sockets: raw counts used as successes and trials
    - genus: categorical (keeps original strings; ensure 'Homo sapiens' exact spelling)
    - tooth_class: categorical with levels preserved
    - age_z: standardized age (z-score)
    - prob_male: kept as-is (0-1 probability)
    - specimen: kept for clustering

    Notes:
    - Drops rows with missing essential information.
    - Ensures sockets > 0. If num_amtl > sockets, num_amtl is truncated to sockets.
    """
    df = df.copy()

    # Required columns check (will raise if missing)
    required_cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")

    # Drop rows missing essential fields
    df = df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen'])

    # Ensure numeric types
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')

    # Drop rows that became NaN after coercion
    df = df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male'])

    # Remove rows with nonpositive socket counts (cannot compute binomial proportion)
    df = df[df['sockets'] > 0]

    # If num_amtl > sockets (possibly data entry), truncate to sockets
    df.loc[df['num_amtl'] > df['sockets'], 'num_amtl'] = df.loc[df['num_amtl'] > df['sockets'], 'sockets']

    # Compute proportion of missing teeth in the recorded sockets
    df['amtl_prop'] = df['num_amtl'] / df['sockets']

    # Standardize age (z-score) for model stability and interpretability
    age_mean = df['age'].mean()
    age_std = df['age'].std(ddof=0) if df['age'].std(ddof=0) != 0 else 1.0
    df['age_z'] = (df['age'] - age_mean) / age_std

    # Ensure categorical columns are of type 'category' with consistent labels
    df['tooth_class'] = df['tooth_class'].astype('category')
    df['genus'] = df['genus'].astype('category')
    df['specimen'] = df['specimen'].astype('category')

    # Keep only columns required for modeling (but keep originals num_amtl/sockets too)
    keep_cols = ['num_amtl', 'sockets', 'amtl_prop', 'genus', 'tooth_class', 'age', 'age_z', 'prob_male', 'specimen']
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a binomial (logit) GLM for AMTL frequency.

    Model specification:
      amtl_prop (proportion) modeled with Binomial family using weights = sockets.
      Predictors: genus (categorical, reference = Pan by default in encoding), tooth_class (categorical), age_z (continuous), prob_male (continuous).

    We compute both conventional and specimen-clustered robust standard errors to account for multiple observations per specimen.

    Returns a dict with keys:
      - 'glm_result': the fitted GLMResults (default covariance)
      - 'glm_result_clustered': the results object with cluster-robust covariances by specimen
      - 'formula': formula used
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    df = df.copy()

    # Basic sanity checks
    if df.shape[0] == 0:
        raise ValueError('Input dataframe is empty after transformation')

    # Formula: model proportion with binomial family and weights = sockets
    # We use C(genus) and C(tooth_class) so categorical encoding is explicit.
    formula = 'amtl_prop ~ C(genus) + C(tooth_class) + age_z + prob_male'

    # Fit GLM with Binomial family; pass weights = sockets so the model treats each row as sockets trials
    model_glm = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), weights=df['sockets'])
    result_glm = model_glm.fit()

    # Compute cluster-robust covariance by specimen to account for within-specimen dependence
    # If specimen has only one observation per specimen, cluster robust will match conventional SEs.
    try:
        result_clustered = result_glm.get_robustcov_results(cov_type='cluster', groups=df['specimen'])
    except Exception:
        # If clustering fails for any reason, return the regular result and note the failure
        result_clustered = None

    # For interpretability, also compute predicted effect for Homo sapiens vs others by estimating
    # marginal effect for genus if desired. The user can inspect result summaries.

    # Return results (caller can print summaries)
    return {
        'formula': formula,
        'glm_result': result_glm,
        'glm_result_clustered': result_clustered
    }


