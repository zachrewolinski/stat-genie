from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/amtl/add_features_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw AMTL dataset into a dataframe ready for binomial GLM.

    Adds/returns columns used in the model:
      - is_human: binary indicator for genus == 'Homo sapiens'
      - amtl_prop: proportion missing = num_amtl / sockets (for use as response with Binomial family and weights=sockets)
      - age_c: centered age (age - mean(age))

    Also filters out invalid rows (missing critical fields, sockets <= 0, num_amtl > sockets).
    """
    # work on a copy
    df = df.copy()

    # Ensure required columns exist
    required = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen']
    missing_cols = [c for c in required if c not in df.columns]
    if len(missing_cols) > 0:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Coerce numeric columns
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')

    # Standard cleaning: remove rows with missing critical data
    df = df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen'])

    # Remove rows with invalid socket counts
    df = df[df['sockets'] > 0]

    # Ensure num_amtl does not exceed sockets; drop implausible rows
    df = df[df['num_amtl'] <= df['sockets']]

    # Create proportion outcome for GLM (response must be in [0,1] when using weights)
    df['amtl_prop'] = df['num_amtl'] / df['sockets']

    # Create binary Homo indicator
    # Use exact match 'Homo sapiens' (as described in schema). If other Homo labels appear, adjust accordingly.
    df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

    # Center age for numerical stability
    df['age_c'] = df['age'] - df['age'].mean()

    # Ensure tooth_class is categorical with sensible categories
    df['tooth_class'] = df['tooth_class'].astype('category')

    # Reset index for clean downstream usage
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial GLM to test whether modern humans have higher AMTL than non-human primates,
    adjusting for age, sex (prob_male), and tooth class. Uses proportion response with weights=sockets.

    Returns the fitted model results with cluster-robust standard errors clustered by specimen.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Check that transform has been applied (required columns present)
    required = ['amtl_prop', 'sockets', 'is_human', 'age_c', 'prob_male', 'tooth_class', 'specimen', 'num_amtl']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataframe missing required columns for modeling: {missing}")

    # Formula: proportion outcome modeled with Binomial family; tooth_class as categorical factor
    formula = 'amtl_prop ~ is_human + age_c + prob_male + C(tooth_class)'

    # Fit GLM with Binomial family, using sockets as weights (proportion response with varying trials)
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), weights=df['sockets'])
    results = model.fit()

    # Compute cluster-robust SE by specimen to account for non-independence of observations from the same individual
    try:
        results_cluster = results.get_robustcov_results(cov_type='cluster', groups=df['specimen'])
    except Exception:
        # Fallback: if clustering fails, return the original results
        results_cluster = results

    # Print a brief summary (users can inspect returned results object for full detail)
    print(results_cluster.summary())

    # Return the cluster-robust results object
    return results_cluster


