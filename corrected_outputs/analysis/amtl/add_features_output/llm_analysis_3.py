from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/amtl/add_features_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Keep only rows with the necessary fields
    required_cols = ['num_amtl', 'sockets', 'genus', 'age', 'prob_male', 'tooth_class', 'specimen']
    df = df.dropna(subset=required_cols)

    # Ensure sockets > 0 (cannot model binomial with zero trials)
    df = df[df['sockets'] > 0].copy()

    # Make sure numeric columns are numeric
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')

    # Drop rows that became NA after coercion
    df = df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male'])

    # Cap num_amtl to be within [0, sockets] to avoid impossible counts
    df['num_amtl'] = df['num_amtl'].clip(lower=0)
    # Ensure integer counts (if floats present due to import), but keep as numeric
    # Use floor for safety if floats are near integers
    df['num_amtl'] = df['num_amtl'].round().astype(int)
    df['sockets'] = df['sockets'].round().astype(int)
    df.loc[df['num_amtl'] > df['sockets'], 'num_amtl'] = df.loc[df['num_amtl'] > df['sockets'], 'sockets']

    # Compute AMTL rate column (useful for diagnostics/plots)
    df['AMTL_rate'] = df['num_amtl'] / df['sockets']

    # Create main independent variable: indicator for Homo sapiens
    # Be robust to whitespace/capitalization
    df['GenusHuman'] = (df['genus'].astype(str).str.strip().str.lower() == 'homo sapiens').astype(int)

    # Center age to improve interpretability/stability
    df['age_c'] = df['age'] - df['age'].mean()

    # Ensure tooth_class and specimen are categorical
    df['tooth_class'] = df['tooth_class'].astype('category')
    df['specimen'] = df['specimen'].astype('category')

    # Return dataframe containing the variables used in modeling (plus some extras useful for diagnostics)
    keep_cols = list(df.columns)  # return full df but transform ensures necessary columns exist
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Formula: model AMTL as binomial counts (num_amtl successes out of sockets trials)
    # Primary predictor: GenusHuman. Controls: centered age, estimated male probability, tooth class.
    # Cluster standard errors by specimen to account for multiple tooth-class rows per individual.

    # Fit binomial GLM using proportion (num_amtl / sockets) with frequency/observation weights = sockets
    formula = 'I(num_amtl / sockets) ~ GenusHuman + age_c + prob_male + C(tooth_class)'

    # Use statsmodels formula-based GLM. We pass weights=sockets so that the binomial model uses sockets as number of trials.
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), weights=df['sockets'])

    # Fit with cluster-robust standard errors clustered on specimen
    # Note: some statsmodels versions accept cov_type in fit(); this uses that interface.
    try:
        results = model.fit(cov_type='cluster', cov_kwds={'groups': df['specimen']})
    except Exception:
        # Fallback: fit normally then get clustered robust covariance via get_robustcov_results
        res = model.fit()
        results = res.get_robustcov_results(cov_type='cluster', groups=df['specimen'])

    # The fitted object 'results' contains regression coefficients on logit scale.
    # Return the full results object so the caller can inspect params, pvalues, confidence intervals, etc.
    return results


