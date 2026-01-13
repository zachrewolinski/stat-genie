from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/replace_with_rvs_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw AMTL dataset into the modeling dataframe.

    Produces the following additional/cleaned columns used by the model:
    - IsHuman: binary indicator (1 if genus == 'Homo sapiens', else 0)
    - amtl_prop: proportion of missing teeth in the scored sockets (num_amtl / sockets)
    - age_c: centered age (age - median(age))
    - ProbMale: renamed prob_male for clarity (kept continuous 0-1)

    Rows with missing critical values or invalid socket counts are dropped.
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Standardize column names (if necessary) and check required columns
    required = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    # Drop rows with missing key variables
    df = df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen'])

    # Ensure numeric types where appropriate
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')

    # Drop rows created as NaN by coercion
    df = df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male'])

    # Remove rows with non-positive sockets or impossible values
    df = df[df['sockets'] > 0]

    # Clamp num_amtl to integer within [0, sockets]
    df['num_amtl'] = df['num_amtl'].clip(lower=0)
    df.loc[df['num_amtl'] > df['sockets'], 'num_amtl'] = df.loc[df['num_amtl'] > df['sockets'], 'sockets']
    # Ensure integer counts
    df['num_amtl'] = df['num_amtl'].round().astype(int)
    df['sockets'] = df['sockets'].round().astype(int)

    # Create proportion variable for GLM (Binomial family uses proportion with trial counts as variance weights)
    df['amtl_prop'] = df['num_amtl'] / df['sockets']

    # Create primary independent variable: IsHuman (1 if Homo sapiens, else 0)
    # Be robust to possible variants of the genus string
    df['IsHuman'] = df['genus'].astype(str).str.strip().str.lower().apply(lambda x: 1 if x in ['homo sapiens', 'homo', 'homo_sapiens', 'human', 'homo sapiens '] else 0)

    # Rename prob_male to ProbMale for model clarity
    df['ProbMale'] = df['prob_male']

    # Center age to improve interpretability and reduce collinearity
    median_age = df['age'].median()
    df['age_c'] = df['age'] - median_age

    # Ensure tooth_class is categorical and clean common whitespace / capitalization
    df['tooth_class'] = df['tooth_class'].astype(str).str.strip().str.capitalize()

    # Keep only rows with reasonable tooth_class values (if dataset contains other unexpected labels)
    allowed_classes = ['Anterior', 'Posterior', 'Premolar']
    df = df[df['tooth_class'].isin(allowed_classes)]

    # Keep index contiguous
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a binomial (logistic) GLM for proportion missing teeth (AMTL) comparing humans to non-humans
    while controlling for tooth class, age, and sex-probability. Cluster-robust standard errors
    by specimen are returned to account for multiple observations per specimen.

    Returns a dictionary with:
      - 'glm_result': the (naive) fitted GLM result object
      - 'glm_result_clustered': the same result with cluster-robust covariance (by specimen)
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Required columns (ensure they exist)
    req = ['amtl_prop', 'sockets', 'IsHuman', 'tooth_class', 'age_c', 'ProbMale', 'specimen', 'num_amtl']
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Transformed dataframe is missing required columns for modeling: {missing}")

    # Formula: proportion missing ~ IsHuman + tooth class + centered age + prob_male
    # We use the proportion (amtl_prop) as the response and pass the number of trials (sockets)
    formula = 'amtl_prop ~ IsHuman + C(tooth_class) + age_c + ProbMale'

    # Fit GLM with Binomial family; supply variance weights as number of trials (sockets)
    # For binomial with proportion endog, var_weights = number of trials is appropriate.
    glm_model = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), var_weights=df['sockets'])
    glm_result = glm_model.fit()

    # Obtain cluster-robust covariance (cluster on specimen to account for repeated measures)
    try:
        glm_result_clustered = glm_result.get_robustcov_results(cov_type='cluster', groups=df['specimen'])
    except Exception:
        # If robust covariance fails (rare), fall back to the original result
        glm_result_clustered = glm_result

    # Provide summary strings and the fitted results objects
    out = {
        'glm_result': glm_result,
        'glm_result_clustered': glm_result_clustered,
        'summary': glm_result.summary().as_text(),
        'summary_clustered': glm_result_clustered.summary().as_text()
    }

    return out


