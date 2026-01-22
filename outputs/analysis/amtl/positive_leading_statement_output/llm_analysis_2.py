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
    Prepare the dataset for binomial (logistic) regression of AMTL frequency.

    Produces the following columns used in the model:
      - AMTL_count: integer number of antemortem missing teeth (from 'num_amtl')
      - Sockets: integer number of observable sockets (from 'sockets')
      - AMTL_rate: AMTL_count / Sockets (proportion, used as response with freq_weights=Sockets)
      - is_human: binary indicator (1 if genus indicates Homo sapiens, 0 otherwise)
      - age_c: centered age (age - mean(age))
      - prob_male: left as-is (0-1 estimate of male)
      - tooth_class: categorical (kept as provided)
      - specimen: kept as provided (for clustering)
      - pop: kept as provided (control)

    Rows with missing essential data or invalid sockets are dropped.
    """
    df = df.copy()

    # Standardize column names if needed (assume input matches schema)
    # Drop rows missing required columns
    required = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen', 'pop']
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError(f"Input dataframe is missing required columns: {missing_required}")

    # Drop rows with NA in core fields
    df = df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen'])

    # Ensure integer counts and valid sockets
    df['AMTL_count'] = df['num_amtl'].astype(float).round().astype(int)
    df['Sockets'] = df['sockets'].astype(float).round().astype(int)

    # Remove impossible rows (non-positive sockets) and ensure counts are within [0, Sockets]
    df = df[df['Sockets'] > 0].copy()
    df.loc[df['AMTL_count'] < 0, 'AMTL_count'] = 0
    df.loc[df['AMTL_count'] > df['Sockets'], 'AMTL_count'] = df.loc[df['AMTL_count'] > df['Sockets'], 'Sockets']

    # Proportion response for binomial regression
    df['AMTL_rate'] = df['AMTL_count'] / df['Sockets']

    # Binary human indicator: robust to capitalization / small variations
    df['is_human'] = df['genus'].astype(str).str.lower().str.contains('homo').astype(int)

    # Center age for numerical stability
    df['age_c'] = df['age'] - df['age'].mean()

    # Keep prob_male as provided; if missing, impute with sample mean (but we dropped NA above)
    df['prob_male'] = df['prob_male'].astype(float)

    # Ensure tooth_class is categorical with expected levels (but leave any unexpected levels as-is)
    df['tooth_class'] = df['tooth_class'].astype('category')

    # Keep specimen and pop as-is (categorical identifiers)
    df['specimen'] = df['specimen'].astype(str)
    df['pop'] = df['pop'].astype('category')

    # Add weights column equal to number of trials (sockets) for clarity
    df['weights'] = df['Sockets']

    # Return only the columns required for modeling plus a few useful identifiers
    cols_kept = ['AMTL_count', 'Sockets', 'AMTL_rate', 'is_human', 'age_c', 'prob_male', 'tooth_class', 'specimen', 'pop', 'weights', 'genus']
    return df[cols_kept]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a binomial (logistic) regression to test whether modern humans (is_human==1)
    have higher AMTL frequency than non-human primates, controlling for age, sex, and tooth class.

    Method:
      - Use GLM with Binomial family and logit link.
      - Model the response as the proportion AMTL_rate with frequency weights equal to Sockets.
      - Include is_human as the main predictor and control for age_c, prob_male, and tooth_class.
      - Compute Odds Ratios (OR) and 95% confidence intervals by exponentiating coefficients.
      - Perform a likelihood-ratio-style comparison between the full model and a reduced model without is_human
        to assess whether adding the human indicator significantly improves fit.
      - Cluster-robust standard errors by specimen (reported alongside the main results).

    Returns a dictionary with the fitted result object, readable summary, OR table, and LR test.
    """
    import statsmodels.api as sm
    import numpy as np
    from scipy import stats

    # Check expected columns
    expected = ['AMTL_rate', 'weights', 'is_human', 'age_c', 'prob_male', 'tooth_class', 'specimen']
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Transformed dataframe is missing required columns for modeling: {missing}")

    # Formula: proportion response modeled with freq_weights = Sockets
    formula = 'AMTL_rate ~ is_human + age_c + prob_male + C(tooth_class)'

    # Fit full model
    glm_full = sm.GLM.from_formula(formula, data=df, family=sm.families.Binomial(), freq_weights=df['weights'])
    res_full = glm_full.fit()

    # Obtain cluster-robust covariance (cluster by specimen)
    try:
        cov_cluster = res_full.get_robustcov_results(cov_type='cluster', groups=df['specimen'])
    except Exception:
        # If clustering fails, fall back to the original result
        cov_cluster = res_full

    # Prepare ORs and CIs (from clustered cov if available)
    params = cov_cluster.params
    conf = cov_cluster.conf_int()
    pvalues = cov_cluster.pvalues

    or_table = pd.DataFrame({
        'coef': params,
        'OR': np.exp(params),
        'ci_lower': np.exp(conf[0]),
        'ci_upper': np.exp(conf[1]),
        'pvalue': pvalues
    })

    # Fit reduced model without is_human for LR-type test
    formula_reduced = 'AMTL_rate ~ age_c + prob_male + C(tooth_class)'
    glm_reduced = sm.GLM.from_formula(formula_reduced, data=df, family=sm.families.Binomial(), freq_weights=df['weights'])
    res_reduced = glm_reduced.fit()

    # Likelihood-ratio statistic (2 * delta log-likelihood). For large samples, compare to chi-square with df=1
    lr_stat = 2.0 * (res_full.llf - res_reduced.llf)
    lr_df = res_full.df_model - res_reduced.df_model
    lr_pvalue = stats.chi2.sf(lr_stat, lr_df)

    results = {
        'full_model_result': res_full,
        'clustered_results': cov_cluster,
        'or_table': or_table,
        'lr_test': {
            'lr_stat': float(lr_stat),
            'df': float(lr_df),
            'pvalue': float(lr_pvalue)
        },
        'summary_text': res_full.summary().as_text()
    }

    return results


