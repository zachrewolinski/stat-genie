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
    Clean and prepare the dataset for binomial regression of AMTL.

    Produces the following explicit columns used in the model:
    - num_amtl: integer number of missing teeth (kept from input)
    - sockets: integer number of observable sockets (kept from input)
    - amtl_prop: proportion num_amtl / sockets (dependent variable for glm)
    - genus: cleaned genus string (categorical independent variable)
    - age_c: centered age (control)
    - prob_male: probability of male (control)
    - tooth_class: categorical tooth class (control)
    """
    df = df.copy()

    # Required input columns
    required_cols = ['num_amtl', 'sockets', 'genus', 'age', 'prob_male', 'tooth_class']
    missing_req = [c for c in required_cols if c not in df.columns]
    if missing_req:
        raise ValueError(f"Input dataframe missing required columns: {missing_req}")

    # Drop rows with missing required fields
    df = df.dropna(subset=required_cols)

    # Ensure numeric/integer types for counts
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')

    # Remove rows where sockets is not positive or counts invalid
    df = df.dropna(subset=['num_amtl', 'sockets'])
    df = df[df['sockets'] > 0]

    # Coerce num_amtl to integer within valid range [0, sockets]
    df['num_amtl'] = df['num_amtl'].round().astype(int)
    df = df[(df['num_amtl'] >= 0) & (df['num_amtl'] <= df['sockets'])]

    # Clean genus and tooth_class text
    df['genus'] = df['genus'].astype(str).str.strip()
    df['tooth_class'] = df['tooth_class'].astype(str).str.strip().str.title()

    # Keep only expected tooth classes (safety) and drop others
    allowed_tooth_classes = {'Anterior', 'Posterior', 'Premolar'}
    df = df[df['tooth_class'].isin(allowed_tooth_classes)]

    # Create proportion and centered age
    df['amtl_prop'] = df['num_amtl'] / df['sockets']
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df = df.dropna(subset=['age'])
    df['age_c'] = df['age'] - df['age'].mean()

    # Ensure prob_male numeric and in [0,1]
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')
    df = df.dropna(subset=['prob_male'])
    df = df[(df['prob_male'] >= 0.0) & (df['prob_male'] <= 1.0)]

    # Convert categorical columns to category dtype for modeling convenience
    df['genus'] = df['genus'].astype('category')
    df['tooth_class'] = df['tooth_class'].astype('category')

    # Final check: non-empty
    if df.shape[0] == 0:
        raise ValueError('No rows remain after cleaning; check input data and filters.')

    # Return only columns needed for modeling (plus helpful ones for diagnostics)
    cols_to_return = ['num_amtl', 'sockets', 'amtl_prop', 'genus', 'age', 'age_c', 'prob_male', 'tooth_class']
    return df[cols_to_return]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a binomial (logistic) GLM to test whether genus predicts AMTL rates
    while controlling for age, sex (prob_male), and tooth class.

    The model uses the proportion amtl_prop as the response and uses sockets
    as frequency weights so that observations with more sockets contribute
    proportionally more information (equivalent to modeling num_amtl with
    a binomial denominator).

    Returns a dictionary with the fitted model object and a table of odds
    ratios (exponentiated coefficients) with 95% CIs and p-values.
    """
    import statsmodels.formula.api as smf

    # Make a safe copy
    df = df.copy()

    # Formula: proportion ~ genus + controls; genus and tooth_class as categorical
    formula = 'amtl_prop ~ C(genus) + age_c + prob_male + C(tooth_class)'

    # Fit GLM with binomial family. Use freq_weights = number of sockets so that
    # model treats amtl_prop as successes / trials (more trials -> more info).
    glm_model = smf.glm(formula=formula,
                        data=df,
                        family=sm.families.Binomial(),
                        freq_weights=df['sockets']).fit()

    # Prepare an odds-ratio table
    params = glm_model.params
    conf_int = glm_model.conf_int()
    pvalues = glm_model.pvalues

    or_table = pd.DataFrame({
        'term': params.index,
        'coef': params.values,
        'OR': np.exp(params.values),
        'CI_lower': np.exp(conf_int[0].values),
        'CI_upper': np.exp(conf_int[1].values),
        'pvalue': pvalues.values
    }).set_index('term')

    results = {
        'model_fit': glm_model,
        'odds_ratios': or_table
    }

    return results


