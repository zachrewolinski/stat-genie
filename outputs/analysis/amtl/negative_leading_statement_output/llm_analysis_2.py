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
    Transform the raw dataset into the analysis dataframe. Produces columns used in modeling:
      - amtl_success: integer count of missing teeth in the scored class
      - amtl_failure: integer count of present/observable teeth (sockets - missing)
      - amtl_prop: proportion missing (amtl_success / sockets)
      - sockets: number of observable sockets used as the binomial denominator
      - is_human: 1 for 'Homo sapiens', 0 otherwise
      - age_c: centered age
      - prob_male: used as provided (0-1)
      - tooth_class: categorical tooth class (kept as-is)
      - specimen: specimen id (categorical) used for clustering
      - pop: population/locality (kept as-is)

    The function drops rows with missing critical values and ensures counts are consistent.
    """
    df = df.copy()

    # Ensure critical columns exist
    required = ['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus', 'specimen']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input dataframe: {missing}")

    # Drop rows missing critical fields
    df = df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus', 'specimen'])

    # Ensure sockets are positive integers and num_amtl is non-negative integer
    # Coerce to numeric and drop invalid rows
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df = df.dropna(subset=['sockets', 'num_amtl'])

    # Keep only rows with at least one observable socket
    df = df[df['sockets'] > 0].copy()

    # Round counts to nearest integer and enforce bounds: 0 <= num_amtl <= sockets
    df['sockets'] = df['sockets'].round().astype(int)
    df['num_amtl'] = df['num_amtl'].round().astype(int)
    # Clip num_amtl to valid range
    df['num_amtl'] = df['num_amtl'].clip(lower=0, upper=df['sockets'])

    # Create outcome counts for binomial model
    df['amtl_success'] = df['num_amtl']
    df['amtl_failure'] = df['sockets'] - df['amtl_success']

    # Proportion (for GEE or diagnostics)
    df['amtl_prop'] = df['amtl_success'] / df['sockets']

    # Indicator for modern human (Homo sapiens). Use exact label present in data: 'Homo sapiens'
    df['is_human'] = (df['genus'].astype(str).str.strip() == 'Homo sapiens').astype(int)

    # Center age for interpretability
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    age_mean = df['age'].mean()
    df['age_c'] = df['age'] - age_mean

    # Ensure prob_male is numeric and bounded to [0,1]
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce').clip(0.0, 1.0)

    # Ensure categorical columns have correct dtype
    df['tooth_class'] = df['tooth_class'].astype('category')
    df['specimen'] = df['specimen'].astype('category')
    if 'pop' in df.columns:
        df['pop'] = df['pop'].astype('category')

    # Final drop in case conversions introduced NAs
    df = df.dropna(subset=['amtl_success', 'amtl_failure', 'age_c', 'prob_male', 'tooth_class', 'specimen'])

    # Keep only columns necessary for modeling + helpful diagnostics
    keep_cols = ['amtl_success', 'amtl_failure', 'amtl_prop', 'sockets', 'is_human', 'age_c', 'prob_male', 'tooth_class', 'specimen']
    if 'pop' in df.columns:
        keep_cols.append('pop')

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit binomial regression(s) to test whether modern humans have higher AMTL rates than non-human primates,
    controlling for age, sex (prob_male), and tooth class. Two complementary approaches are provided:
      1) GLM (Binomial) with cluster-robust standard errors clustered by specimen
      2) GEE (Binomial) with exchangeable correlation within specimen (accounts for repeated measures per specimen)

    Returns a dictionary with model result objects and a focused odds-ratio inference for the is_human effect.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import numpy as np

    results = {}

    # Check required columns
    req = ['amtl_success', 'amtl_failure', 'is_human', 'age_c', 'prob_male', 'tooth_class', 'specimen', 'sockets']
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for modeling: {missing}")

    # Prepare design matrix: use one-hot (drop-first) for tooth_class
    tooth_dummies = pd.get_dummies(df['tooth_class'], prefix='tooth', drop_first=True)
    exog = pd.concat([df[['is_human', 'age_c', 'prob_male']].reset_index(drop=True), tooth_dummies.reset_index(drop=True)], axis=1)
    exog = sm.add_constant(exog, has_constant='add')

    # Endog as two-column (successes, failures) for Binomial in statsmodels
    endog = np.column_stack((df['amtl_success'].values, df['amtl_failure'].values))

    # 1) GLM Binomial with cluster-robust SEs by specimen
    glm_model = sm.GLM(endog, exog, family=sm.families.Binomial())
    try:
        glm_res = glm_model.fit()
        # Recompute parameter covariance clustered by specimen
        clustered_res = glm_res.get_robustcov_results(cov_type='cluster', groups=df['specimen'])
        results['glm_clustered'] = clustered_res
    except Exception as e:
        # Fallback: return the non-robust fit if clustering fails
        glm_res = glm_model.fit()
        results['glm_clustered'] = glm_res
        results['glm_clustered_note'] = f"Clustered covariance computation raised: {e}. Returning non-robust fit."

    # 2) GEE (Binomial) grouping by specimen using proportion outcome with weights = sockets
    # Prepare proportion endog and weights
    df_gee = df.reset_index(drop=True).copy()
    df_gee['amtl_prop'] = df_gee['amtl_success'] / df_gee['sockets']

    formula = 'amtl_prop ~ is_human + age_c + prob_male + C(tooth_class)'
    try:
        gee_model = smf.gee(formula, groups='specimen', data=df_gee, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable(), weights=df_gee['sockets'])
        gee_res = gee_model.fit()
        results['gee'] = gee_res
    except Exception as e:
        results['gee_error'] = str(e)

    # Focused inference for the is_human effect from the clustered GLM (preferred for coefficient interpretation):
    try:
        coef_name = 'is_human'
        coef = results['glm_clustered'].params[coef_name]
        se = results['glm_clustered'].bse[coef_name]
        z = coef / se
        pval = results['glm_clustered'].pvalues[coef_name]
        ci_lower, ci_upper = results['glm_clustered'].conf_int().loc[coef_name]
        # Convert to odds ratio scale
        or_est = np.exp(coef)
        or_ci_lower, or_ci_upper = np.exp([ci_lower, ci_upper])

        results['is_human_inference'] = {
            'coef_logit': float(coef),
            'se': float(se),
            'z': float(z),
            'pvalue': float(pval),
            'ci_logit': [float(ci_lower), float(ci_upper)],
            'odds_ratio': float(or_est),
            'odds_ratio_ci': [float(or_ci_lower), float(or_ci_upper)]
        }
    except Exception as e:
        results['is_human_inference_error'] = str(e)

    # Attach model summaries (text) for quick inspection
    try:
        results['glm_summary'] = results['glm_clustered'].summary().as_text()
    except Exception:
        results['glm_summary'] = None
    try:
        if 'gee' in results:
            results['gee_summary'] = results['gee'].summary().as_text()
    except Exception:
        results['gee_summary'] = None

    # The calling code can inspect results['is_human_inference'] to answer the research question:
    # If odds_ratio > 1 and pvalue < 0.05, that would support humans having higher AMTL rates after controls.

    return results


