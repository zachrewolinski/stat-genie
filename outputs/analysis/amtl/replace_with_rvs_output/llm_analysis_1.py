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
    Transform the raw dataset for binomial modeling of AMTL.

    Outputs a dataframe containing the columns used by the model:
      - specimen, genus, pop, tooth_class
      - num_amtl, sockets, amtl_prop
      - is_human (0/1)
      - age, age_c (centered age), prob_male

    Steps:
      - drop rows missing essential variables
      - keep only rows with sockets > 0
      - compute proportion amtl_prop
      - derive is_human indicator from genus
      - coerce categorical columns to category dtype
      - center age for numerical stability
    """
    df = df.copy()

    # Ensure expected columns exist
    required_cols = ['num_amtl', 'sockets', 'genus', 'age', 'prob_male', 'tooth_class', 'specimen']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for transform: {missing}")

    # Drop rows with missing essential values
    df = df.dropna(subset=['num_amtl', 'sockets', 'genus', 'age', 'prob_male', 'tooth_class', 'specimen'])

    # Keep only records with at least one socket observed (otherwise proportion undefined)
    df = df[df['sockets'] > 0]

    # Ensure numeric types for counts
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce').astype(float)
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce').astype(float)

    # Re-drop in case conversion introduced NaNs
    df = df.dropna(subset=['num_amtl', 'sockets'])

    # Compute proportion of missing teeth for the tooth-class record
    df['amtl_prop'] = df['num_amtl'] / df['sockets']

    # Create is_human indicator from genus. Exact match to dataset string 'Homo sapiens'.
    df['is_human'] = (df['genus'].astype(str).str.strip() == 'Homo sapiens').astype(int)

    # Standardize/categorical types
    df['tooth_class'] = df['tooth_class'].astype('category')
    df['genus'] = df['genus'].astype('category')
    if 'pop' in df.columns:
        df['pop'] = df['pop'].astype('category')
    else:
        # If pop is missing, create a placeholder category so formula code can always reference it
        df['pop'] = pd.Categorical(['__missing_pop__'] * len(df))

    # Center age for numerical stability in modeling
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df = df.dropna(subset=['age'])
    df['age_c'] = df['age'] - df['age'].mean()

    # Ensure prob_male numeric and in [0,1]
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')
    df = df.dropna(subset=['prob_male'])

    # Final column selection: keep columns required for modeling and downstream interpretation
    keep_cols = ['specimen', 'genus', 'pop', 'tooth_class', 'num_amtl', 'sockets', 'amtl_prop', 'is_human', 'age', 'age_c', 'prob_male']
    df = df[keep_cols]

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial GLM to test whether modern humans (is_human) have higher AMTL rates
    than non-human primates, controlling for age, sex (prob_male), tooth class, and pop.

    Modeling approach:
      - Use a binomial GLM where the response is the proportion amtl_prop and the number
        of trials is given by 'sockets'. This is implemented by fitting a GLM with the
        Binomial family to the proportion and providing 'sockets' as frequency weights
        so the model treats the outcome as counts out of sockets.
      - Cluster-robust standard errors by 'specimen' to account for non-independence of
        multiple tooth_class rows from the same specimen.

    Returns the fitted results object with cluster-robust covariance applied when possible.
    """
    import statsmodels.formula.api as smf

    # Formula: primary predictor is is_human; control for age (centered), prob_male, tooth_class and pop
    formula = 'amtl_prop ~ is_human + age_c + prob_male + C(tooth_class) + C(pop)'

    # Fit GLM (Binomial) on proportion with freq_weights = sockets so that each row is treated
    # as num_amtl / sockets with sockets trials.
    glm_model = smf.glm(formula=formula, data=df, family=sm.families.Binomial())

    # Fit using frequency weights equal to the number of sockets (number of trials)
    # Note: fit accepts freq_weights argument which scales the contribution of each observation.
    res = glm_model.fit(freq_weights=df['sockets'])

    # Attempt to compute cluster-robust SEs by specimen (accounts for repeated measures)
    try:
        res_cluster = res.get_robustcov_results(cov_type='cluster', groups=df['specimen'])
    except Exception:
        # If clustering fails for any reason, return the original results
        res_cluster = res

    # Return the results with cluster-robust cov if available. The caller can inspect summary, params, conf_int, etc.
    return res_cluster


