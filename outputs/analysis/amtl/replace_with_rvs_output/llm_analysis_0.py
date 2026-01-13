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
    Transform the raw AMTL dataset into a dataframe ready for binomial regression.

    Produces the following additional columns used in modeling:
    - prop_amtl : proportion of missing teeth (num_amtl / sockets)
    - IsHuman   : binary indicator (1 if genus == 'Homo sapiens', else 0)
    - age_c     : age centered and standardized
    - prob_male_c: prob_male centered and standardized
    - tooth_class: categorical (kept as category dtype)

    Rows with missing critical fields or with sockets <= 0 are removed.
    """
    df = df.copy()

    # Required columns for analysis
    required = ['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus', 'specimen']
    # Drop rows missing any required fields
    df = df.dropna(subset=required)

    # Ensure sockets are positive integers > 0
    df = df[df['sockets'] > 0].copy()

    # Compute proportion of AMTL for binomial modeling (used with weights = sockets)
    df['prop_amtl'] = df['num_amtl'].astype(float) / df['sockets'].astype(float)

    # Create binary indicator for modern humans (Homo sapiens)
    # allow for minor variations in genus string (strip and lower)
    df['IsHuman'] = (df['genus'].astype(str).str.strip().str.lower() == 'homo sapiens').astype(int)

    # Center and standardize age and prob_male for better numeric stability
    df['age_c'] = (df['age'] - df['age'].mean()) / (df['age'].std(ddof=0) if df['age'].std(ddof=0) != 0 else 1.0)
    df['prob_male_c'] = (df['prob_male'] - df['prob_male'].mean()) / (df['prob_male'].std(ddof=0) if df['prob_male'].std(ddof=0) != 0 else 1.0)

    # Ensure tooth_class is categorical. If 'Posterior' is present, set it as the first category so it acts as the reference by default.
    try:
        cats = list(df['tooth_class'].dropna().unique())
        # if Posterior present, put it first
        if 'Posterior' in cats:
            ordered = ['Posterior'] + [c for c in cats if c != 'Posterior']
            df['tooth_class'] = pd.Categorical(df['tooth_class'], categories=ordered)
        else:
            df['tooth_class'] = df['tooth_class'].astype('category')
    except Exception:
        df['tooth_class'] = df['tooth_class'].astype('category')

    # Keep only the columns needed for modeling plus a few identifiers
    model_cols = [
        'specimen', 'num_amtl', 'sockets', 'prop_amtl', 'IsHuman',
        'age_c', 'prob_male_c', 'tooth_class', 'genus', 'age', 'prob_male'
    ]
    # Some columns may not exist if input lacked them; guard by intersection
    model_cols = [c for c in model_cols if c in df.columns]

    return df[model_cols].reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial (logistic) regression comparing AMTL rates between modern humans and non-human primates,
    controlling for age, sex (probability male), and tooth class. Cluster-robust standard errors are computed
    at the specimen level to account for multiple observations per specimen.

    Returns the fitted results object with cluster-robust covariance.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Expect the transformed dataframe to contain: prop_amtl, sockets, IsHuman, age_c, prob_male_c, tooth_class, specimen
    required = ['prop_amtl', 'sockets', 'IsHuman', 'age_c', 'prob_male_c', 'tooth_class', 'specimen']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Transformed dataframe is missing required columns for modeling: {missing}")

    # Formula: model the proportion prop_amtl with binomial family and use sockets as weights
    # Include tooth_class as a categorical predictor (statsmodels will create dummies)
    formula = 'prop_amtl ~ IsHuman + age_c + prob_male_c + C(tooth_class)'

    # Fit GLM (Binomial) with weights = number of sockets (denominator)
    glm_binom = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), weights=df['sockets']).fit()

    # Compute cluster-robust covariance (cluster by specimen) to account for within-specimen correlation
    try:
        glm_clus = glm_binom.get_robustcov_results(cov_type='cluster', groups=df['specimen'])
    except Exception:
        # Fallback: if clustering fails, return the original fit
        glm_clus = glm_binom

    return glm_clus


