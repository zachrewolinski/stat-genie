from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/replace_and_positive_statement_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Gilmore (2013) AMTL dataset into a dataframe suitable for binomial regression.

    Output columns used in the model:
      - num_amtl: integer count of missing teeth (kept from original)
      - sockets: integer count of observable sockets (kept from original)
      - IsHuman: binary indicator (1 if genus == 'Homo sapiens', else 0)
      - prop_amtl: proportion missing = num_amtl / sockets (used for GLM with weights)
      - age, age_c: age and mean-centered age
      - prob_male: sex estimate (0-1)
      - tooth_class: categorical (Anterior/Posterior/Premolar)
      - stdev_age: uncertainty in age (filled if missing)
      - pop: categorical population/provenance
      - specimen: specimen identifier (string) for clustering
    """
    df = df.copy()

    # Basic required columns check
    required = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen', 'pop', 'stdev_age']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    # Drop rows missing essential outcome or predictors
    df = df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class'])

    # Keep only rows where sockets > 0 (can't model proportion otherwise)
    df = df[df['sockets'] > 0].copy()

    # Ensure integer counts for binomial model
    df['num_amtl'] = df['num_amtl'].astype(float)
    df['sockets'] = df['sockets'].astype(float)

    # Create binary indicator for Homo sapiens
    # Normalize genus strings to allow 'Homo sapiens' variants
    df['IsHuman'] = (df['genus'].astype(str).str.strip().str.lower() == 'homo sapiens').astype(int)

    # Proportion missing (useful for GLM with weights)
    df['prop_amtl'] = df['num_amtl'] / df['sockets']

    # Center age (improves interpretability and numeric stability)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    mean_age = df['age'].mean()
    df['age_c'] = df['age'] - mean_age

    # Ensure prob_male is numeric and clipped to [0,1]
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce').fillna(0.5)
    df['prob_male'] = df['prob_male'].clip(0.0, 1.0)

    # Fill or clean stdev_age
    df['stdev_age'] = pd.to_numeric(df['stdev_age'], errors='coerce')
    # Replace missing stdev_age with median of available values
    if df['stdev_age'].isnull().any():
        med = df['stdev_age'].median()
        if pd.isnull(med):
            med = 1.0
        df['stdev_age'] = df['stdev_age'].fillna(med)

    # Coerce categorical variables
    df['tooth_class'] = df['tooth_class'].astype('category')
    df['pop'] = df['pop'].astype('category')
    df['specimen'] = df['specimen'].astype(str)

    # Return the transformed dataframe with only columns needed for modeling (and a few helpers)
    cols_to_keep = [
        'num_amtl', 'sockets', 'prop_amtl', 'IsHuman', 'age', 'age_c', 'prob_male',
        'tooth_class', 'stdev_age', 'pop', 'specimen'
    ]
    # Keep any additional columns that were in the original but ensure required ones are present
    for c in cols_to_keep:
        if c not in df.columns:
            df[c] = np.nan

    return df[cols_to_keep]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a binomial GLM to test whether modern humans (IsHuman == 1) have higher AMTL than non-human primates,
    controlling for age (centered), sex (prob_male), tooth class, stdev_age, and population (pop).

    Implementation notes:
      - Uses prop_amtl as the response (num_amtl / sockets) with freq_weights = sockets for binomial counts.
      - Fits a GLM with Binomial family and reports cluster-robust SEs clustered by specimen.
      - Returns the fitted result object with cluster-robust covariance, the coefficient and OR for IsHuman,
        and diagnostics (dispersion).
    """
    df = df.copy()

    # Basic checks
    if df[['num_amtl', 'sockets', 'prop_amtl']].isnull().any().any():
        raise ValueError('Transformed dataframe contains missing values in outcome columns. Ensure transform() was applied and rows with missing outcomes were removed.')

    # Formula: model the probability of a socket being missing
    formula = 'prop_amtl ~ IsHuman + age_c + prob_male + C(tooth_class) + stdev_age + C(pop)'

    # Fit GLM with binomial family using sockets as frequencies/weights
    # Using freq_weights here makes the model treat prop_amtl as proportion with corresponding trials
    model_glm = sm.GLM.from_formula(formula, data=df, family=sm.families.Binomial(), freq_weights=df['sockets'])
    res = model_glm.fit()

    # Obtain cluster-robust standard errors clustered by specimen
    # If clustering fails for any reason, fall back to the original result
    try:
        res_cl = res.get_robustcov_results(cov_type='cluster', groups=df['specimen'])
    except Exception:
        res_cl = res

    # Extract coefficient, SE, OR and CIs for IsHuman
    out = {}
    out['model_result'] = res_cl

    if 'IsHuman' in res_cl.params.index:
        coef = float(res_cl.params['IsHuman'])
        se = float(res_cl.bse['IsHuman'])
        ci_low, ci_high = res_cl.conf_int().loc['IsHuman'].values
        or_est = float(np.exp(coef))
        or_ci = (float(np.exp(ci_low)), float(np.exp(ci_high)))
        out.update({
            'IsHuman_coef': coef,
            'IsHuman_se': se,
            'IsHuman_pvalue': float(res_cl.pvalues['IsHuman']),
            'IsHuman_OR': or_est,
            'IsHuman_OR_CI': or_ci
        })
    else:
        out.update({
            'IsHuman_coef': np.nan,
            'IsHuman_se': np.nan,
            'IsHuman_pvalue': np.nan,
            'IsHuman_OR': np.nan,
            'IsHuman_OR_CI': (np.nan, np.nan)
        })

    # Compute a simple dispersion statistic (Pearson chi-square / df) to check for overdispersion
    # For binomial GLM, dispersion should be ~1. Large >1 suggests overdispersion.
    pearson_chi2 = sum(res.resid_pearson ** 2)
    df_resid = float(res.df_resid) if res.df_resid is not None else np.nan
    dispersion = (pearson_chi2 / df_resid) if df_resid and df_resid > 0 else np.nan
    out['dispersion'] = dispersion
    out['pearson_chi2'] = pearson_chi2
    out['df_resid'] = df_resid

    # Attach a short textual summary for quick inspection
    try:
        out['summary_text'] = res_cl.summary().as_text()
    except Exception:
        out['summary_text'] = None

    return out


