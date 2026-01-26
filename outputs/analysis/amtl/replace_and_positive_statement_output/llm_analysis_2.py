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
    Transform raw Gilmore (2013) AMTL dataset into analysis-ready dataframe.

    Output columns used in modeling:
      - amtl_prop : num_amtl / sockets (float proportion)
      - sockets   : number of observable sockets (int) used as binomial trials (weights)
      - IsHuman   : 1 if genus == 'Homo sapiens', else 0
      - age_z     : standardized age (z-score)
      - prob_male : estimated probability specimen was male (0-1)
      - tooth_class : categorical tooth class (Anterior/Posterior/Premolar)
      - specimen  : specimen identifier (grouping)
    """
    df = df.copy()

    # Keep only required columns and drop rows with missing critical fields
    required = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen']
    df = df.dropna(subset=required)

    # Ensure sockets is positive integer > 0
    df = df[df['sockets'].astype(float) > 0]

    # Ensure num_amtl is non-negative and does not exceed sockets
    # Drop rows where num_amtl is invalid
    df = df[(df['num_amtl'].astype(float) >= 0) & (df['num_amtl'].astype(float) <= df['sockets'].astype(float))]

    # Create AMTL proportion (endog) and keep sockets for binomial trials (weights)
    df['amtl_prop'] = df['num_amtl'].astype(float) / df['sockets'].astype(float)

    # Create binary human indicator
    # Use exact string 'Homo sapiens' per dataset schema
    df['IsHuman'] = (df['genus'].astype(str) == 'Homo sapiens').astype(int)

    # Standardize age to aid model convergence and interpretation
    df['age_z'] = (df['age'].astype(float) - df['age'].astype(float).mean()) / (df['age'].astype(float).std(ddof=0) if df['age'].astype(float).std(ddof=0) != 0 else 1.0)

    # Ensure prob_male is numeric and between 0 and 1
    df['prob_male'] = df['prob_male'].astype(float).clip(0.0, 1.0)

    # Force tooth_class and specimen to categorical dtype for modeling
    df['tooth_class'] = df['tooth_class'].astype('category')
    df['specimen'] = df['specimen'].astype('category')

    # Keep only the columns necessary for the model (but preserve originals for inspection)
    keep_cols = ['num_amtl', 'sockets', 'amtl_prop', 'IsHuman', 'age_z', 'prob_male', 'tooth_class', 'specimen', 'genus', 'pop']
    for c in keep_cols:
        if c not in df.columns:
            # safe-guard: create column of NaNs if missing
            df[c] = np.nan

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a binomial GEE to test whether modern humans (IsHuman == 1) have higher AMTL than non-human primates,
    while adjusting for age_z, prob_male, and tooth_class. Uses specimen as the grouping variable (exchangeable correlation)
    to account for repeated observations per specimen.

    Returns a dictionary with the fitted results object and a summarized table of coefficients, odds ratios, CIs, and p-values.
    """
    # Import the necessary statsmodels pieces
    import statsmodels.api as sm
    from statsmodels.genmod.cov_struct import Exchangeable

    # Defensive checks
    if 'amtl_prop' not in df.columns or 'sockets' not in df.columns:
        raise ValueError("Transformed dataframe must contain 'amtl_prop' and 'sockets' columns")

    # Build and fit GEE with binomial family. Use the proportion as endog and sockets as weights (trials).
    formula = 'amtl_prop ~ IsHuman + age_z + prob_male + C(tooth_class)'

    # Construct GEE model (exchangeable correlation within specimen)
    gee_model = sm.GEE.from_formula(formula,
                                    groups='specimen',
                                    data=df,
                                    family=sm.families.Binomial(),
                                    cov_struct=Exchangeable(),
                                    weights=df['sockets'])

    result = gee_model.fit()

    # Compute odds ratios and 95% CI for interpretability
    params = result.params
    conf = result.conf_int()
    or_est = np.exp(params)
    or_ci_lower = np.exp(conf[0])
    or_ci_upper = np.exp(conf[1])

    summary_table = pd.DataFrame({
        'coef': params,
        'OR': or_est,
        'CI_lower': or_ci_lower,
        'CI_upper': or_ci_upper,
        'pvalue': result.pvalues
    })

    # If the main hypothesis is that humans have higher AMTL, inspect IsHuman coefficient
    # The table provides the evidence (coefficient, OR, p-value). The user can check summary_table.loc['IsHuman']

    return {
        'model_result': result,
        'summary_table': summary_table
    }


