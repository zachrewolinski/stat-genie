from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/amtl/anonymize_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe suitable for binomial regression of AMTL.

    The function:
    - Renames dataset columns to clear analysis names.
    - Ensures numeric types where appropriate and drops rows with missing critical data.
    - Removes rows with zero or negative numbers of observable sockets.
    - Computes proportion missing and a binary IsHuman indicator.
    - Converts genus and tooth_class to categorical types.

    Returns the transformed dataframe which must contain the columns:
    ['specimen_id','n_missing','n_observed','prop_missing','age','age_sd','sex_female','genus','tooth_class','region','IsHuman']
    (some columns are optional for the model but included for completeness).
    """
    df = df.copy()

    # Rename input schema columns to analysis-friendly names
    rename_map = {
        'feature1': 'tooth_class',      # Anterior, Posterior, Premolar
        'feature2': 'specimen_id',      # specimen identifier
        'feature3': 'n_missing',        # number of teeth missing of given class
        'feature4': 'n_observed',       # number of observable sockets for that class
        'feature5': 'age',              # estimated age at death
        'feature6': 'age_sd',           # uncertainty of age estimate
        'feature7': 'sex_female',       # estimate of femaleness (0-1)
        'feature8': 'genus',            # Pan, Pongo, Homo sapiens, Papio
        'feature9': 'region'            # region of origin
    }
    df = df.rename(columns=rename_map)

    # Ensure numeric columns are numeric (coerce errors to NaN)
    numeric_cols = ['n_missing', 'n_observed', 'age', 'age_sd', 'sex_female']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing critical variables for the binomial model
    required = ['n_missing', 'n_observed', 'genus', 'tooth_class', 'age', 'sex_female']
    df = df.dropna(subset=[c for c in required if c in df.columns])

    # Remove rows where there are no observable sockets (can't estimate a proportion)
    df = df[df['n_observed'] > 0].copy()

    # Compute the proportion missing (endogenous for Binomial model when paired with weights)
    df['prop_missing'] = df['n_missing'] / df['n_observed']
    # Ensure proportions are bounded [0,1]
    df['prop_missing'] = df['prop_missing'].clip(0, 1)

    # Binary indicator for modern humans (Homo sapiens)
    # Use a robust string comparison in case of capitalization/whitespace differences
    df['IsHuman'] = (df['genus'].astype(str).str.strip() == 'Homo sapiens').astype(int)

    # Convert categorical columns to pandas categorical dtype for convenience
    df['tooth_class'] = df['tooth_class'].astype('category')
    df['genus'] = df['genus'].astype('category')
    if 'region' in df.columns:
        df['region'] = df['region'].astype('category')

    # Return only the columns necessary for modeling plus identifying columns
    keep_cols = [c for c in ['specimen_id', 'n_missing', 'n_observed', 'prop_missing', 'age', 'age_sd', 'sex_female', 'genus', 'tooth_class', 'region', 'IsHuman'] if c in df.columns]
    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial GLM predicting the probability of a tooth being missing (AMTL) as a function of
    being a modern human (IsHuman), tooth class, age, and sex. The model uses the number of observed
    sockets as binomial trials (weights) and the proportion missing as the response.

    Returns the fitted model object (and, if possible, a clustered-robust covariance variant by region).
    """
    import statsmodels.formula.api as smf

    # Require that the df has the necessary columns
    required = ['prop_missing', 'n_observed', 'IsHuman', 'tooth_class', 'age', 'sex_female']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Dataframe is missing required columns for modeling: {missing}")

    # Formula: model proportion missing with binomial denominator given by n_observed (weights)
    # C(tooth_class) treats tooth_class as categorical. IsHuman is the main predictor of interest.
    formula = 'prop_missing ~ IsHuman + C(tooth_class) + age + sex_female'

    # Fit GLM with Binomial family using weights = number of trials (n_observed)
    glm_binom = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), weights=df['n_observed'])
    fit = glm_binom.fit()

    # If region is present, provide clustered robust standard errors by region as an alternative
    if 'region' in df.columns:
        try:
            clustered = fit.get_robustcov_results(cov_type='cluster', groups=df['region'])
            # Return both the standard fit and the clustered-robust variant
            return {'fit': fit, 'clustered_fit_by_region': clustered}
        except Exception:
            # If clustering fails for any reason, return the plain fit
            return fit

    return fit


