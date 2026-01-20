from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/amtl/noperturb_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw AMTL dataset into a dataframe ready for binomial GLM.
    Produces the following columns used in the model:
      - amtl_rate: num_amtl / sockets (proportion)
      - sockets: number of observable sockets (used as binomial denominator / var_weights)
      - IsHuman: 1 if genus == 'Homo sapiens', else 0
      - age_c: centered age (age - mean(age))
      - prob_male_c: centered prob_male (prob_male - mean(prob_male))
      - tooth_class_Premolar, tooth_class_Posterior: dummy indicators (reference: Anterior)
      - pop, specimen: retained identifiers (pop used for clustered SEs)
    """
    df = df.copy()

    # Drop rows that cannot be used in a binomial model: need non-missing num_amtl, sockets, and positive sockets
    df = df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus'])
    df = df[df['sockets'] > 0]

    # AMTL rate (proportion) -- dependent variable for binomial model
    df['amtl_rate'] = df['num_amtl'].astype(float) / df['sockets'].astype(float)

    # Main IV: indicator for modern human
    # Normalize genus text then create indicator
    df['genus_str'] = df['genus'].astype(str).str.strip()
    df['IsHuman'] = (df['genus_str'].str.lower() == 'homo sapiens').astype(int)
    df = df.drop(columns=['genus_str'])

    # Center continuous covariates to aid interpretation
    df['age_c'] = df['age'].astype(float) - df['age'].astype(float).mean()
    df['prob_male_c'] = df['prob_male'].astype(float) - df['prob_male'].astype(float).mean()

    # Tooth class dummies: keep Premolar and Posterior; use Anterior as reference
    tooth_dummies = pd.get_dummies(df['tooth_class'].astype(str), prefix='tooth_class')
    # Preferably drop 'tooth_class_Anterior' to set it as reference; if absent drop the first column
    if 'tooth_class_Anterior' in tooth_dummies.columns:
        tooth_dummies = tooth_dummies.drop(columns=['tooth_class_Anterior'])
    elif tooth_dummies.shape[1] > 0:
        tooth_dummies = tooth_dummies.drop(columns=[tooth_dummies.columns[0]])

    # Ensure both expected columns exist in final df (create if needed with zeros)
    for col in ['tooth_class_Premolar', 'tooth_class_Posterior']:
        if col not in tooth_dummies.columns:
            tooth_dummies[col] = 0

    # Concatenate dummies to dataframe (only the two columns of interest retained)
    df = pd.concat([df.reset_index(drop=True), tooth_dummies[['tooth_class_Premolar', 'tooth_class_Posterior']].reset_index(drop=True)], axis=1)

    # Keep identifier columns used later
    if 'pop' not in df.columns:
        df['pop'] = pd.NA
    if 'specimen' not in df.columns:
        df['specimen'] = pd.NA

    # Final selected columns for modeling (kept in df for downstream use)
    keep_cols = ['specimen', 'pop', 'num_amtl', 'sockets', 'amtl_rate', 'IsHuman', 'age_c', 'prob_male_c', 'tooth_class_Premolar', 'tooth_class_Posterior']
    # Some datasets might have extra columns; ensure keep columns exist
    for c in keep_cols:
        if c not in df.columns:
            df[c] = pd.NA

    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit a binomial (logistic) GLM for AMTL proportion using the transformed dataframe.

    Model specification (on log-odds scale):
      amtl_rate ~ IsHuman + age_c + prob_male_c + tooth_class_Premolar + tooth_class_Posterior

    The model uses 'sockets' as observation weights (binomial denominator) so that each row's
    variance is weighted appropriately. Robust clustered standard errors by 'pop' are computed
    to account for non-independence within populations.

    Returns a dict with:
      - 'model': the fitted statsmodels results object with clustered SEs (if available)
      - 'odds_ratios': exponentiated coefficients
      - 'odds_ratio_ci': exponentiated 95% CI for coefficients
    """
    df = df.copy()

    # Ensure predictor columns exist (if transform produced them, they should be present)
    predictors = ['IsHuman', 'age_c', 'prob_male_c', 'tooth_class_Premolar', 'tooth_class_Posterior']
    for p in predictors:
        if p not in df.columns:
            # If missing, create a zero column (safe fallback)
            df[p] = 0.0

    # Drop rows with missing outcome or weights
    df = df.dropna(subset=['amtl_rate', 'sockets'])

    # Design matrix
    X = df[predictors]
    X = sm.add_constant(X, has_constant='add')

    # Fit binomial GLM for proportion with var_weights equal to sockets (the binomial denominator)
    glm = sm.GLM(df['amtl_rate'].astype(float), X.astype(float), family=sm.families.Binomial(), var_weights=df['sockets'].astype(float))
    res = glm.fit()

    # Compute clustered robust covariance by population 'pop' if available
    results = res
    try:
        # If 'pop' has NA or is entirely unique, clustering may fail; guard with try/except
        clustered = res.get_robustcov_results(cov_type='cluster', groups=df['pop'])
        results = clustered
    except Exception:
        # Fall back to original model results if clustering fails
        clustered = res
        results = res

    # Exponentiated coefficients (odds ratios) and CI
    params = results.params
    conf = results.conf_int()
    odds_ratios = np.exp(params)
    odds_ratio_ci = np.exp(conf)

    # Print a short summary to help interpret results when running interactively
    try:
        print(results.summary())
    except Exception:
        pass

    return {
        'model': results,
        'odds_ratios': odds_ratios,
        'odds_ratio_ci': odds_ratio_ci
    }


