from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/add_features_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe ready for binomial GLM.

    - Drops rows with missing essential values
    - Removes observations with non-positive sockets or logically invalid counts
    - Standardizes genus and tooth_class strings
    - Creates amtl_rate (num_amtl / sockets)
    - Centers age to improve model fitting (age_c)

    Returns the transformed dataframe containing at minimum the columns:
    ['num_amtl','sockets','amtl_rate','genus','tooth_class','age','age_c','prob_male']
    """
    df = df.copy()

    # Required columns
    required_cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns in input dataframe: {missing}")

    # Drop rows with missing essential data
    df = df.dropna(subset=required_cols)

    # Ensure numeric types where expected
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')

    # Drop any rows that became NA after coercion
    df = df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male'])

    # Only observations with at least one scored socket make sense
    df = df[df['sockets'] > 0]

    # Ensure num_amtl is integer and within [0, sockets]
    # Round to nearest integer if non-integer values present
    df['num_amtl'] = df['num_amtl'].round().astype(int)
    df = df[(df['num_amtl'] >= 0) & (df['num_amtl'] <= df['sockets'])]

    # Standardize string columns
    df['genus'] = df['genus'].astype(str).str.strip()
    df['tooth_class'] = df['tooth_class'].astype(str).str.strip().str.capitalize()

    # Create proportion outcome
    df['amtl_rate'] = df['num_amtl'] / df['sockets']

    # Center age to help model convergence and interpretability
    df['age_c'] = df['age'] - df['age'].mean()

    # Keep only tooth_class levels that are expected (safety) and drop any unexpected levels
    expected_tooth_classes = set(['Anterior', 'Posterior', 'Premolar'])
    df = df[df['tooth_class'].isin(expected_tooth_classes)]

    # Optionally, standardize genus names for common variants (e.g., 'Homo' -> 'Homo sapiens')
    # This attempts to make sure human specimens are labeled consistently. If no mapping applies, keep original.
    df['genus'] = df['genus'].replace({
        'Homo': 'Homo sapiens',
        'H. sapiens': 'Homo sapiens'
    })

    # Final dropna safety for new columns
    df = df.dropna(subset=['amtl_rate', 'age_c'])

    # Return the prepared dataframe
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a binomial GLM to test whether genus (especially Homo sapiens) predicts higher AMTL rates
    while controlling for age, sex (prob_male), and tooth class.

    Returns a dictionary containing:
    - 'glm_results': the fitted GLM results (default covariance)
    - 'robust_results': the fitted GLM results with robust (HC3) covariance
    - 'overdispersion': estimated overdispersion statistic (Pearson chi2 / df_resid)

    Model specification (on proportions with weights = sockets):
    amtl_rate ~ C(genus) + age_c + prob_male + C(tooth_class)
    family = Binomial, weights = sockets (number of trials per observation)
    """
    import statsmodels.formula.api as smf

    # Make a local copy to avoid modifying input
    df = df.copy()

    # Ensure required columns are present
    req = ['amtl_rate', 'num_amtl', 'sockets', 'genus', 'age_c', 'prob_male', 'tooth_class']
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Formula: proportion response with genus and covariates
    formula = 'amtl_rate ~ C(genus) + age_c + prob_male + C(tooth_class)'

    # Fit binomial GLM using sockets as freq_weights (number of trials per observation)
    glm_model = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), freq_weights=df['sockets'])
    res = glm_model.fit()

    # Compute overdispersion: Pearson chi-square / df_resid
    try:
        pearson_chi2 = (res.resid_pearson ** 2).sum()
        df_resid = res.df_resid
        overdispersion = pearson_chi2 / df_resid if df_resid > 0 else float('nan')
    except Exception:
        pearson_chi2 = float('nan')
        overdispersion = float('nan')

    # Robust covariance results (HC3) for inference robust to heteroskedasticity / mild misspecification
    try:
        robust_res = res.get_robustcov_results(cov_type='HC3')
    except Exception:
        robust_res = None

    # Return results and diagnostic info
    return {
        'glm_results': res,
        'robust_results': robust_res,
        'pearson_chi2': pearson_chi2,
        'df_resid': df_resid,
        'overdispersion': overdispersion,
        'formula': formula
    }


