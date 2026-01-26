from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/positive_leading_statement_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Gilmore (2013) dataset for binomial GLM analysis of AMTL.

    Produces the following columns used by the model:
      - amtl: integer number of antemortem tooth losses (successes)
      - sockets: integer number of observable sockets (trials)
      - non_amtl: integer number of non-missing teeth (sockets - amtl)
      - is_human: 1 if genus == 'Homo sapiens', else 0
      - age_c: centered age (age - mean(age))
      - age_sq: (age_c)^2
      - prob_male: numeric probability specimen is male (clipped to [0,1])
      - tooth_Anterior: dummy (1 if tooth_class == 'Anterior')
      - tooth_Premolar: dummy (1 if tooth_class == 'Premolar')
      - pop: population / provenance (kept for clustering or fixed effects)
      - specimen: specimen id (kept for traceability)

    Rows with missing or invalid essential values are dropped.
    """
    df = df.copy()

    # Required columns
    required = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'pop', 'specimen']
    df = df.dropna(subset=required)

    # Ensure numeric and valid ranges
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')

    df = df.dropna(subset=['sockets', 'num_amtl', 'age', 'prob_male'])

    # Remove impossible values
    df = df[df['sockets'] >= 1]
    df = df[(df['num_amtl'] >= 0) & (df['num_amtl'] <= df['sockets'])]

    # Binary indicator for modern humans
    df['is_human'] = (df['genus'].astype(str).str.strip() == 'Homo sapiens').astype(int)

    # Center age and include quadratic term to capture non-linear effects
    df['age_c'] = df['age'] - df['age'].mean()
    df['age_sq'] = df['age_c'] ** 2

    # Clip prob_male to [0,1] in case of small numeric issues
    df['prob_male'] = df['prob_male'].clip(0.0, 1.0)

    # Tooth-class dummies: use Posterior as implicit reference (both dummies = 0)
    df['tooth_Anterior'] = (df['tooth_class'] == 'Anterior').astype(int)
    df['tooth_Premolar'] = (df['tooth_class'] == 'Premolar').astype(int)

    # Successes and failures for binomial model
    df['amtl'] = df['num_amtl'].astype(int)
    df['non_amtl'] = (df['sockets'] - df['num_amtl']).astype(int)

    # Keep only columns needed for modeling and tracing
    out_cols = [
        'specimen', 'amtl', 'non_amtl', 'sockets',
        'is_human', 'age', 'age_c', 'age_sq', 'prob_male',
        'tooth_Anterior', 'tooth_Premolar', 'genus', 'pop'
    ]

    return df[out_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a binomial GLM for AMTL with the following specification:
      endog = (amtl, sockets - amtl)
      exog = const + is_human + age_c + age_sq + prob_male + tooth_Anterior + tooth_Premolar

    Returns a dictionary with the fitted model and a version with cluster-robust SEs by population (pop).
    """
    # Build binomial endog as (successes, failures)
    endog = np.vstack([df['amtl'].values, (df['sockets'].values - df['amtl'].values)]).T

    # Exogenous variables
    exog_cols = ['is_human', 'age_c', 'age_sq', 'prob_male', 'tooth_Anterior', 'tooth_Premolar']
    exog = df[exog_cols].astype(float)
    exog = sm.add_constant(exog, has_constant='add')

    # Fit GLM (binomial)
    glm_binom = sm.GLM(endog, exog, family=sm.families.Binomial())
    res = glm_binom.fit()

    # Compute cluster-robust covariance by population (pop) if available
    clustered_res = None
    if 'pop' in df.columns:
        try:
            clustered_res = res.get_robustcov_results(cov_type='cluster', groups=df['pop'])
        except Exception:
            # Some grouping schemes may fail (e.g., too many small groups). In that case, leave None.
            clustered_res = None

    # Return the raw results objects so the user can inspect coefficients, CIs, p-values, etc.
    return {
        'glm_result': res,
        'glm_result_clustered_by_pop': clustered_res
    }


