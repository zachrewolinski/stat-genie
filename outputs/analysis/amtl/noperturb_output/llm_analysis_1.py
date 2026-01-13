from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/noperturb_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the dataframe used for modeling. Required output columns:
      - num_amtl: integer count of missing teeth for the tooth class
      - sockets: integer count of observable sockets (trials)
      - prop_missing: num_amtl / sockets (proportion)
      - age, age_c: estimated age and centered age
      - prob_male, prob_male_c: estimated probability male and centered
      - genus: taxon (string/categorical)
      - tooth_class: tooth class (string/categorical)
      - specimen: specimen ID (string)

    Steps:
      - drop rows missing essential fields
      - coerce numeric columns, drop rows with sockets <= 0
      - remove impossible rows where num_amtl > sockets
      - compute proportion and center continuous covariates
      - ensure genus and tooth_class are strings (categorical)
    """
    # Work on a copy
    df = df.copy()

    # Required columns check and drop missing
    required = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen']
    df = df.dropna(subset=required)

    # Coerce numeric columns
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')

    # Drop rows where sockets is missing or not positive
    df = df[df['sockets'].notnull()]
    df = df[df['sockets'] > 0]

    # Remove impossible counts (more missing teeth than observable sockets)
    df = df[df['num_amtl'].notnull()]
    df = df[df['num_amtl'] <= df['sockets']]
    df = df[df['num_amtl'] >= 0]

    # Compute proportion missing (will be used as response); keep raw counts for binomial formulation
    df['prop_missing'] = df['num_amtl'] / df['sockets']

    # Sanitize prob_male to [0,1]
    df['prob_male'] = df['prob_male'].clip(0.0, 1.0)

    # Center continuous covariates for interpretability/stability
    df['age_c'] = df['age'] - df['age'].mean()
    df['prob_male_c'] = df['prob_male'] - df['prob_male'].mean()

    # Ensure categorical/string types for genus and tooth_class
    df['genus'] = df['genus'].astype(str)
    df['tooth_class'] = df['tooth_class'].astype(str)
    df['specimen'] = df['specimen'].astype(str)

    # Normalize genus naming if common variants exist (non-destructive mapping)
    # If dataset uses 'Homo' instead of 'Homo sapiens', map it to 'Homo sapiens'
    df['genus'] = df['genus'].replace({'Homo': 'Homo sapiens'})

    # Final selection of columns used in modeling
    out_cols = ['num_amtl', 'sockets', 'prop_missing', 'age', 'age_c', 'prob_male', 'prob_male_c', 'genus', 'tooth_class', 'specimen']
    df = df.loc[:, out_cols]

    # Reset index for cleanliness
    df = df.reset_index(drop=True)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial regression comparing AMTL rates across genera while adjusting for age, sex (probability male), and tooth class.

    Modeling approach:
      - Use a GLM with binomial family where the response is the proportion prop_missing and weights = sockets (trials)
      - Include genus as a categorical predictor with 'Homo sapiens' as the reference level
      - Adjust for centered age (age_c), centered prob_male (prob_male_c), and tooth_class (categorical)
      - Compute cluster-robust standard errors clustered by specimen to account for non-independence of observations from the same specimen

    Returns the fitted results object with cluster-robust covariances.
    """
    # Ensure inputs exist
    required = ['num_amtl', 'sockets', 'prop_missing', 'age_c', 'prob_male_c', 'genus', 'tooth_class', 'specimen']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataframe missing required columns for modeling: {missing}")

    # Define model formula: compare genera to Homo sapiens (reference)
    # Using Patsy's Treatment coding to set the reference group
    formula = 'prop_missing ~ C(genus, Treatment(reference="Homo sapiens")) + age_c + prob_male_c + C(tooth_class)'

    # Fit GLM (binomial) with freq_weights equal to number of trials (sockets)
    # Using proportions as endog and weights = sockets implements a binomial with varying trials
    glm_model = sm.GLM.from_formula(formula,
                                   data=df,
                                   family=sm.families.Binomial(),
                                   freq_weights=df['sockets'])
    res = glm_model.fit()

    # Obtain cluster-robust standard errors clustered by specimen
    # This adjusts standard errors for within-specimen non-independence
    try:
        res_cluster = res.get_robustcov_results(cov_type='cluster', groups=df['specimen'])
    except Exception:
        # If robust clustering fails for any reason, return the original fit but warn the user
        import warnings
        warnings.warn('Cluster-robust covariance computation failed; returning nominal GLM results.')
        res_cluster = res

    # Print a brief summary and return the clustered-robust results object
    print(res_cluster.summary())
    return res_cluster


