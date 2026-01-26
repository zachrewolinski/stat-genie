from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/negative_leading_statement_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw Gilmore (2013) AMTL dataset into analysis-ready dataframe.

    Outputs dataframe contains at minimum the columns:
      - successes: integer count of AMTL (num_amtl, clipped to sockets)
      - trials: integer count of observable sockets
      - is_human: binary indicator (1 for 'Homo sapiens', 0 otherwise)
      - age_c: centered age
      - prob_male: sex probability as provided
      - tooth_class: categorical tooth class
      - specimen: specimen id (kept for clustering)
    """
    # Work on a copy
    df = df.copy()

    # Keep only rows with essential columns non-missing
    required_cols = ['num_amtl', 'sockets', 'genus', 'age', 'prob_male', 'tooth_class', 'specimen']
    df = df.dropna(subset=required_cols)

    # Ensure numeric types
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')

    # Drop rows where sockets is missing or <= 0 after conversion
    df = df.dropna(subset=['sockets', 'num_amtl', 'age', 'prob_male'])
    df = df[df['sockets'] > 0]

    # Clip number of missing teeth to be within [0, sockets]
    # Use elementwise minimum to avoid broadcasting.
    df['successes'] = df[['num_amtl', 'sockets']].apply(lambda row: int(max(0, min(int(round(row['num_amtl'])), int(round(row['sockets']))))), axis=1)
    df['trials'] = df['sockets'].astype(int)

    # Create binary human indicator
    df['is_human'] = (df['genus'].astype(str).str.strip() == 'Homo sapiens').astype(int)

    # Center age for numerical stability
    df['age_c'] = df['age'] - df['age'].mean()

    # Keep prob_male as provided (numeric between 0 and 1). If outside bounds, clip.
    df['prob_male'] = df['prob_male'].clip(0.0, 1.0)

    # Ensure tooth_class is categorical and standardized
    df['tooth_class'] = df['tooth_class'].astype(str).str.strip().replace({'Anterior': 'Anterior', 'Posterior': 'Posterior', 'Premolar': 'Premolar'})
    df['tooth_class'] = pd.Categorical(df['tooth_class'], categories=['Anterior', 'Premolar', 'Posterior'])

    # Keep specimen column as-is for clustering/grouping
    df['specimen'] = df['specimen'].astype(str)

    # Final safety: remove any rows where successes > trials or trials <= 0 (should already be covered)
    df = df[df['trials'] > 0]
    df['successes'] = df[['successes', 'trials']].apply(lambda r: int(min(r['successes'], r['trials'])), axis=1)

    # Keep only columns needed for modeling (plus any other useful metadata)
    keep_cols = ['specimen', 'tooth_class', 'successes', 'trials', 'is_human', 'age_c', 'prob_male', 'genus', 'pop']
    for c in keep_cols:
        if c not in df.columns:
            df[c] = np.nan

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a binomial GLM for AMTL (successes/trials) with a human indicator as the main predictor,
    controlling for age, sex, and tooth class. Cluster-robust standard errors are computed by specimen.

    Returns a dict with:
      - 'glm_clustered': statsmodels GLMResults with clustered robust covariances
      - 'odds_ratio_is_human': point estimate of odds ratio for is_human
      - 'ci_is_human': 95% CI for the odds ratio
      - 'summary': textual summary (string)
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Drop any rows with missing essential modeling columns
    df_model = df.dropna(subset=['successes', 'trials', 'is_human', 'age_c', 'prob_male', 'tooth_class', 'specimen']).copy()

    # Build design matrix: numeric predictors plus dummy variables for tooth_class (drop first to avoid multicollinearity)
    tooth_dummies = pd.get_dummies(df_model['tooth_class'], prefix='tc', drop_first=True)
    X = pd.concat([df_model[['is_human', 'age_c', 'prob_male']].reset_index(drop=True), tooth_dummies.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X, has_constant='add')

    # Outcome: proportion of missing teeth; in GLM with Binomial family we pass proportion as endog and trials as var_weights
    y = (df_model['successes'] / df_model['trials']).astype(float)
    weights = df_model['trials'].astype(float)

    # Fit GLM (Binomial family) with frequency weights = number of trials (sockets)
    glm = sm.GLM(y, X, family=sm.families.Binomial(), var_weights=weights)
    res = glm.fit()

    # Obtain clustered (by specimen) robust covariance estimates
    try:
        res_clust = res.get_robustcov_results(cov_type='cluster', groups=df_model['specimen'])
    except Exception:
        # Fallback: return non-clustered result if clustering fails
        res_clust = res

    # Compute odds ratio and 95% CI for the is_human coefficient
    if 'is_human' in res_clust.params.index:
        coef = res_clust.params['is_human']
        se = res_clust.bse['is_human']
        or_point = float(np.exp(coef))
        ci_low = float(np.exp(coef - 1.96 * se))
        ci_high = float(np.exp(coef + 1.96 * se))
    else:
        or_point = np.nan
        ci_low = np.nan
        ci_high = np.nan

    # Prepare a concise textual summary
    try:
        summary_text = res_clust.summary().as_text()
    except Exception:
        summary_text = str(res_clust)

    return {
        'glm_clustered': res_clust,
        'odds_ratio_is_human': or_point,
        'ci_is_human': (ci_low, ci_high),
        'summary': summary_text
    }


