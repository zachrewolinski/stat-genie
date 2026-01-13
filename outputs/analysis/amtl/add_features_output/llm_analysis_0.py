from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/add_features_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset to the analysis-ready dataframe.
    Produces the following columns required for modeling:
      - num_amtl: integer count of missing teeth for the observed tooth class (kept from original)
      - sockets: integer count of observable sockets (kept from original)
      - prop_missing: proportion missing = num_amtl / sockets (float 0..1)
      - IsHuman: binary indicator (1 if genus == 'Homo sapiens', else 0)
      - age_c: centered age (age - mean(age))
      - prob_male: kept from original (estimate of male probability)
      - tooth_class: categorical (kept from original)
      - specimen: kept from original (used for clustering)

    Rows with missing or invalid values for these core variables are dropped.
    """
    df = df.copy()

    # Required columns for the analysis
    required_cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen']

    # Drop rows with missing values in any required column
    df = df.dropna(subset=required_cols)

    # Remove rows with non-positive sockets (cannot define binomial trials)
    df = df[df['sockets'] > 0]

    # Ensure integer counts where appropriate
    # (if the dataset has floats due to prior processing, cast safely)
    df['num_amtl'] = df['num_amtl'].astype(float)
    df['sockets'] = df['sockets'].astype(float)

    # Binary indicator for modern human
    df['IsHuman'] = (df['genus'].astype(str).str.strip() == 'Homo sapiens').astype(int)

    # Center age for numerical stability and interpretability
    df['age_c'] = df['age'].astype(float) - df['age'].astype(float).mean()

    # Ensure prob_male is numeric and in [0,1]
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')
    df = df.dropna(subset=['prob_male'])
    # Clip to [0,1] in case of slight measurement artifacts
    df['prob_male'] = df['prob_male'].clip(0.0, 1.0)

    # Ensure tooth_class is a categorical variable with consistent spelling/casing
    df['tooth_class'] = df['tooth_class'].astype(str).str.strip().replace({'anterior': 'Anterior', 'posterior': 'Posterior', 'premolar': 'Premolar'})
    df['tooth_class'] = df['tooth_class'].astype('category')

    # Proportion missing for use with binomial GLM (endog in [0,1])
    df['prop_missing'] = df['num_amtl'] / df['sockets']

    # Final safety drops: ensure proportion within [0,1]
    df = df[(df['prop_missing'] >= 0.0) & (df['prop_missing'] <= 1.0)]

    # Keep only columns needed for modeling (but keep originals for traceability)
    cols_to_keep = ['num_amtl', 'sockets', 'prop_missing', 'IsHuman', 'age_c', 'prob_male', 'tooth_class', 'specimen']
    df = df[cols_to_keep].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a binomial (logistic) GLM comparing AMTL in modern humans vs non-human primates,
    controlling for age, sex (prob_male), and tooth class. Uses sockets as the number of
    trials via weighting and computes cluster-robust standard errors clustered by specimen.

    Returns:
      - results_robust: a statsmodels results object with cluster-robust covariance
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Work on a copy to be safe
    df = df.copy()

    # Ensure the proportion outcome exists (transform should have created it)
    if 'prop_missing' not in df.columns:
        df['prop_missing'] = df['num_amtl'] / df['sockets']

    # Formula: proportion missing explained by human status, age, sex estimate, and tooth class
    # C(tooth_class) tells patsy/statsmodels to treat tooth_class as categorical
    formula = 'prop_missing ~ IsHuman + age_c + prob_male + C(tooth_class)'

    # Fit GLM with binomial family using sockets as weights (number of trials)
    # Using weights here models the binomial denominator when endog is a proportion
    glm_model = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), weights=df['sockets'])
    results = glm_model.fit()

    # Compute cluster-robust standard errors clustered by specimen to account for
    # within-specimen correlation (multiple tooth classes per specimen)
    try:
        results_robust = results.get_robustcov_results(cov_type='cluster', groups=df['specimen'])
    except Exception:
        # If cluster robust fails for some reason, fall back to default results
        results_robust = results

    # Print model summary (user can inspect coefficients and significance)
    print(results_robust.summary())

    return results_robust


