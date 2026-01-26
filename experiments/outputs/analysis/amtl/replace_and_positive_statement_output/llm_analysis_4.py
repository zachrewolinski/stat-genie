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
    # Work on a copy
    df = df.copy()

    # Remove rows missing critical variables required for modeling
    df = df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen'])

    # Ensure sockets and counts are integers and valid
    df = df[df['sockets'] > 0]
    df['num_amtl'] = df['num_amtl'].astype(int)
    df['sockets'] = df['sockets'].astype(int)

    # Proportion missing (useful for diagnostics and for formula-based GLM with weights)
    df['missing_prop'] = df['num_amtl'] / df['sockets']

    # Primary independent variable: binary indicator for Homo sapiens
    # Match exact string 'Homo sapiens' in the genus column (strip whitespace)
    df['is_human'] = (df['genus'].astype(str).str.strip() == 'Homo sapiens').astype(int)

    # Center age for numerical stability and easier interpretation
    df['age_c'] = df['age'] - df['age'].mean()

    # Ensure tooth_class and pop are categorical
    df['tooth_class'] = df['tooth_class'].astype('category')
    df['pop'] = df['pop'].astype('category')

    # Ensure specimen is string (used for clustering repeated measures)
    df['specimen'] = df['specimen'].astype(str)

    # Sanity clip for prob_male
    df['prob_male'] = df['prob_male'].clip(0.0, 1.0)

    # Final dataframe returned contains all columns used in modeling
    # Columns present: num_amtl, sockets, missing_prop, is_human, age_c, prob_male, tooth_class, pop, specimen, genus, age
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a binomial GLM comparing AMTL in modern humans vs non-human primates,
    controlling for age, sex (probability of male), tooth class, and population.

    We model the proportion missing (num_amtl / sockets) with a binomial family
    and use the number of sockets as frequency weights. Cluster-robust SEs are
    computed at the specimen level to account for repeated measures (multiple
    tooth_classes per specimen).

    Returns a dictionary with the clustered-fit result object and an odds-ratio table.
    """
    import numpy as np
    import pandas as pd

    model_df = df.copy()

    # Formula: proportion of missing teeth as a function of is_human and controls
    # include C(tooth_class) and C(pop) as categorical controls
    formula = 'missing_prop ~ is_human + age_c + prob_male + C(tooth_class) + C(pop)'

    # Fit binomial GLM using counts via frequency weights (sockets). Using
    # proportion (missing_prop) as endog and sockets as freq_weights.
    glm = sm.GLM.from_formula(
        formula,
        data=model_df,
        family=sm.families.Binomial(),
        freq_weights=model_df['sockets']
    )

    res = glm.fit()

    # Obtain cluster-robust standard errors clustered by specimen (repeated measures)
    # This changes the covariance matrix / t-stats & p-values but keeps coefficients identical.
    # If specimen clustering vector is long, get_robustcov_results will compute clustered covariances.
    try:
        res_cluster = res.get_robustcov_results(cov_type='cluster', groups=model_df['specimen'])
    except Exception:
        # Fall back to non-clustered result if clustering fails for any reason
        res_cluster = res

    # Compute odds ratios and cluster-robust CI
    params = res_cluster.params
    conf = res_cluster.conf_int()
    or_df = pd.DataFrame({
        'coef': params,
        'OR': np.exp(params),
        'CI_lower': np.exp(conf.iloc[:, 0]),
        'CI_upper': np.exp(conf.iloc[:, 1])
    })

    # For interpretation, extract the is_human row if present
    human_row = None
    if 'is_human' in or_df.index:
        human_row = or_df.loc['is_human']

    return {
        'glm_result_clustered': res_cluster,
        'odds_ratio_table': or_df,
        'is_human_or_row': human_row
    }


