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
    Transform the raw dataset into the analysis-ready dataframe.

    Produces:
    - amtl_prop: proportion of missing teeth (num_amtl / sockets)
    - is_human: binary indicator for Homo specimens (1) vs non-human (0)
    - age_c: age centered around the sample mean
    - prob_male_c: prob_male centered around the sample mean
    - ensures tooth_class and genus are categorical

    Drops rows with missing critical values and rows with sockets <= 0.
    """
    df = df.copy()

    # Required columns present in the dataset schema
    required_cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class']
    # Drop rows missing any required columns
    df = df.dropna(subset=required_cols)

    # Remove rows where sockets is zero or negative (cannot form proportion / binomial trials)
    df = df[df['sockets'] > 0]

    # Ensure numeric types where appropriate
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')

    # Re-drop rows that became NaN after coercion
    df = df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male'])

    # Create proportion column (successes / trials) for binomial modeling
    df['amtl_prop'] = df['num_amtl'] / df['sockets']

    # Create binary human indicator. Match 'Homo' (case-insensitive) in genus.
    # Use .astype(str) guard in case genus is categorical
    df['is_human'] = df['genus'].astype(str).str.lower().str.contains('homo').astype(int)

    # Center continuous covariates to improve interpretability and model stability
    df['age_c'] = df['age'] - df['age'].mean()
    df['prob_male_c'] = df['prob_male'] - df['prob_male'].mean()

    # Ensure tooth_class and genus are categorical with consistent labels
    df['tooth_class'] = df['tooth_class'].astype('category')
    df['genus'] = df['genus'].astype('category')

    # Optional: drop any rows where amtl_prop is outside [0,1] due to data issues
    df = df[(df['amtl_prop'] >= 0) & (df['amtl_prop'] <= 1)]

    # Return the dataframe containing all columns needed for modeling
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a binomial (logit) regression to test whether modern humans have higher AMTL than non-human primates,
    controlling for age, sex (prob_male), and tooth class.

    Modeling approach:
    - Use the proportion amtl_prop as the response and supply freq_weights = sockets so the model treats
      the data as num_amtl successes out of sockets trials (Binomial).
    - Primary parameter of interest: coefficient for is_human (1 = Homo, 0 = non-human).
    - Controls: age_c (centered age), prob_male_c (centered sex estimate), C(tooth_class) (categorical).

    Returns a dictionary with the fitted model, summary, and an odds-ratio table with 95% CIs.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Build formula: proportion outcome modeled with freq_weights = sockets (trials)
    formula = 'amtl_prop ~ is_human + age_c + prob_male_c + C(tooth_class)'

    # Fit GLM with Binomial family; use freq_weights to indicate number of trials per observation
    # (this treats amtl_prop as successes/trials = num_amtl / sockets)
    glm_model = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), freq_weights=df['sockets']).fit()

    # Create a table with exponentiated coefficients (odds ratios) and 95% CIs
    params = glm_model.params
    bse = glm_model.bse
    conf = glm_model.conf_int()
    or_table = pd.DataFrame({
        'coef': params,
        'se': bse,
        'z_or_t': glm_model.tvalues,
        'pvalue': glm_model.pvalues,
        'OR': np.exp(params),
        'OR_CI_lower': np.exp(conf[0]),
        'OR_CI_upper': np.exp(conf[1])
    })

    results = {
        'model': glm_model,
        'summary': glm_model.summary(),
        'or_table': or_table
    }

    return results


