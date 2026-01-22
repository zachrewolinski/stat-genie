from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/panda_nuts/noperturb_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the chimpanzee nut-cracking dataset for modeling.

    Produces the following new/clean columns used by the model:
      - age_z: standardized age (mean 0, sd 1)
      - sex_M: 1 if male, 0 if female (maps other/unrecognized to NaN and such rows are dropped)
      - help_Y: 1 if help indicated (y/yes/etc.), 0 otherwise
      - log_exposure: natural log of session duration in seconds (offset for rate model)
      - hammer: cleaned categorical hammer column (string)
      - chimpanzee: integer ID of individual

    Also drops rows with missing essential fields or non-positive session durations.
    """
    df = df.copy()

    # Drop rows missing essential columns for analysis
    required = ['nuts_opened', 'seconds', 'age', 'sex', 'help', 'hammer', 'chimpanzee']
    df = df.dropna(subset=required)

    # Ensure numeric columns have appropriate dtype
    df['nuts_opened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Drop rows where conversion introduced NaNs
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age'])

    # Ensure session duration is positive (needed for exposure/offset)
    df = df[df['seconds'] > 0]

    # Standardize age (z-score) for model stability
    df['age_z'] = (df['age'] - df['age'].mean()) / (df['age'].std(ddof=0) if df['age'].std(ddof=0) != 0 else 1.0)

    # Clean and map sex to binary (male=1, female=0)
    df['sex'] = df['sex'].astype(str).str.lower().str.strip()
    df['sex_M'] = df['sex'].map({'m': 1, 'male': 1, 'f': 0, 'female': 0})

    # Clean and map help to binary (yes=1 else 0)
    df['help'] = df['help'].astype(str).str.lower().str.strip()
    df['help_Y'] = df['help'].apply(lambda x: 1 if x in ['y', 'yes', 'true', '1'] else 0)

    # Clean hammer values to strings (leave categorical handling to model formula C(hammer))
    df['hammer'] = df['hammer'].astype(str).str.strip()

    # Ensure chimpanzee ID is integer for clustering/grouping
    # If chimpanzee IDs are not numeric, keep as-is (string); but prefer integer if possible
    try:
        df['chimpanzee'] = pd.to_numeric(df['chimpanzee'], errors='coerce').astype(pd.Int64Dtype())
        # If conversion produced NA for some, coerce back to original string representation for those
        na_mask = df['chimpanzee'].isna()
        if na_mask.any():
            df.loc[na_mask, 'chimpanzee'] = df.loc[na_mask, 'chimpanzee'].astype(object)
    except Exception:
        # leave chimpanzee as original if conversion fails
        df['chimpanzee'] = df['chimpanzee'].astype(str)

    # Exposure / offset: log of seconds
    df['log_exposure'] = np.log(df['seconds'])

    # Final drop of rows where sex_M is missing (unrecognized sex labels)
    df = df.dropna(subset=['sex_M'])

    # Reset index
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a negative-binomial GLM modeling the count of nuts opened with session duration as exposure.

    Model specification (formula):
      nuts_opened ~ age_z + sex_M + help_Y + C(hammer)
    Family: NegativeBinomial (accounts for overdispersion relative to Poisson)
    Offset: log_exposure (log seconds) to model the nut-opening rate per second.

    We compute cluster-robust standard errors by chimpanzee to account for within-individual correlation
    of multiple sessions. The returned object is the GLM results with clustered robust covariance.
    """
    import statsmodels.formula.api as smf

    # Ensure required columns exist
    required = ['nuts_opened', 'age_z', 'sex_M', 'help_Y', 'hammer', 'log_exposure', 'chimpanzee']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Build formula: model count with categorical hammer
    formula = 'nuts_opened ~ age_z + sex_M + help_Y + C(hammer)'

    # Fit Negative Binomial GLM with offset = log_exposure
    model_glm = smf.glm(formula=formula,
                        data=df,
                        family=sm.families.NegativeBinomial(),
                        offset=df['log_exposure'])

    res = model_glm.fit()

    # Obtain cluster-robust standard errors clustered by chimpanzee ID
    # If chimpanzee is numeric, use it directly; otherwise use its values for grouping
    try:
        groups = df['chimpanzee']
        res_cluster = res.get_robustcov_results(cov_type='cluster', groups=groups)
    except Exception:
        # Fallback to the original results if clustering fails
        res_cluster = res

    # Print a concise summary (coefs, SEs, z, p)
    print(res_cluster.summary())

    return res_cluster


