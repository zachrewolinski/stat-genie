from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/replace_with_rvs_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw contest-level dataframe into the analysis-ready dataframe.

    Steps performed:
    - Drop rows with missing values in variables needed for the model.
    - Ensure numeric types where appropriate.
    - Create raw difference variables for group size and distances.
    - Standardize the two main predictors (z-score) to aid interpretation and model convergence:
        - RelGroupSize: z-scored (n_focal - n_other)
        - HomeAdvantage: z-scored (dist_other - dist_focal)
    - Create male/female difference control variables (m_diff, f_diff).
    - Ensure win is integer 0/1.

    Returns the transformed dataframe containing at least the columns:
    ['win', 'RelGroupSize', 'HomeAdvantage', 'RelGroupSize_raw', 'HomeAdvantage_raw', 'm_diff', 'f_diff', 'dyad', 'focal', 'other']
    """
    df = df.copy()

    # Required columns for the planned analysis
    required = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other',
                'm_focal', 'm_other', 'f_focal', 'f_other', 'dyad', 'focal', 'other']

    # Drop rows with missing data in required columns
    df = df.dropna(subset=required)

    # Ensure numeric types
    numeric_cols = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other',
                    'm_focal', 'm_other', 'f_focal', 'f_other', 'dyad', 'focal', 'other']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # After coercion, drop any rows that became NA
    df = df.dropna(subset=numeric_cols)

    # Make sure win is integer 0/1
    df['win'] = df['win'].astype(int)

    # Raw difference measures
    df['RelGroupSize_raw'] = df['n_focal'] - df['n_other']
    df['HomeAdvantage_raw'] = df['dist_other'] - df['dist_focal']

    # Controls: differences in male and female counts
    df['m_diff'] = df['m_focal'] - df['m_other']
    df['f_diff'] = df['f_focal'] - df['f_other']

    # Standardize the two main predictors (z-score). Use population std (ddof=0) for stable scaling.
    # If the std is zero (constant column), leave the standardized column as raw (avoids division by zero).
    for raw_col, z_col in [('RelGroupSize_raw', 'RelGroupSize'), ('HomeAdvantage_raw', 'HomeAdvantage')]:
        mean = df[raw_col].mean()
        std = df[raw_col].std(ddof=0)
        if std == 0 or np.isnan(std):
            # fallback: copy raw if no variation
            df[z_col] = df[raw_col]
        else:
            df[z_col] = (df[raw_col] - mean) / std

    # Keep only the columns we will need downstream (but return full df with these present)
    # (The model function will accept the returned df and uses the specified columns.)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a binomial (logistic) generalized linear model to predict the probability
    that the focal group wins a contest.

    Model specification:
      win ~ RelGroupSize * HomeAdvantage + m_diff + f_diff

    - RelGroupSize: standardized difference in group size (focal - other)
    - HomeAdvantage: standardized (dist_other - dist_focal): positive => focal closer to its center
    - Interaction included to test whether the effect of relative group size depends on location/home advantage
    - m_diff and f_diff are included as controls

    Robust inference:
    - After fitting the GLM, cluster-robust standard errors by 'dyad' are computed
      to account for repeated contests between the same pair of groups.

    Returns a dictionary with keys:
      - 'model' : the clustered-results statsmodels object (so you can inspect params, pvalues, conf_int)
      - 'odds_ratios' : pandas DataFrame with exponentiated coefficients and clustered CIs
      - 'glm_result' : the original GLM fit object (useful if desired)
    """
    import statsmodels.formula.api as smf

    # Ensure required columns exist
    required = ['win', 'RelGroupSize', 'HomeAdvantage', 'm_diff', 'f_diff', 'dyad']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Formula with interaction
    formula = 'win ~ RelGroupSize * HomeAdvantage + m_diff + f_diff'

    # Fit GLM (logistic)
    glm_fit = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Cluster-robust covariance by dyad
    # If dyad is numeric, pass it directly; statsmodels will group by identical values.
    try:
        clustered = glm_fit.get_robustcov_results(cov_type='cluster', groups=df['dyad'])
    except Exception:
        # If clustering fails for some reason, fall back to the default results
        clustered = glm_fit

    # Summarize effect sizes as odds ratios with clustered CIs
    params = clustered.params
    conf = clustered.conf_int()
    # conf is a DataFrame-like with two columns [lower, upper]
    or_series = np.exp(params)
    or_ci_lower = np.exp(conf.iloc[:, 0])
    or_ci_upper = np.exp(conf.iloc[:, 1])

    or_table = pd.DataFrame({
        'OR': or_series,
        'OR_2.5%': or_ci_lower,
        'OR_97.5%': or_ci_upper,
        'pvalue': clustered.pvalues
    })

    # Print brief summaries to console for convenience (optional)
    print("GLM (binomial) with clustered SE by dyad:\n")
    print(clustered.summary())
    print('\nOdds ratios (with clustered 95% CI):\n')
    print(or_table)

    return {'model': clustered, 'odds_ratios': or_table, 'glm_result': glm_fit}


