from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/amtl/shuffle_names_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a cleaned dataframe containing the variables used in modeling.

    Assumptions based on provided schema (fields appear to be mislabeled):
    - 'stdev_age' column contains the number of teeth missing for the given tooth class (i.e., AMTL count).
    - 'prob_male' column contains the number of observable sockets (i.e., trials for binomial).
    - 'num_amtl' column contains the estimated age at death (years).
    - 'pop' column contains an estimated probability of being male (0-1).
    - 'age' column actually contains the specimen genus (e.g., 'Homo sapiens', 'Pan', 'Pongo', 'Papio').
    - 'genus' column actually contains the tooth class (e.g., 'Anterior', 'Posterior', 'Premolar').

    The function creates the following final columns used in the model:
      - num_missing : number of missing teeth (AMTL) [float/ints]
      - n_sockets   : number of observable sockets (trials) [int]
      - age_years   : estimated age at death [numeric]
      - sex_prob_male : estimated probability male [0-1]
      - tooth_class : tooth class as categorical
      - genus       : specimen genus (categorical)
      - IsHuman     : binary indicator (1 if genus indicates Homo/human, else 0)
      - AMTL_rate   : num_missing / n_sockets (proportion)
    """

    df = df.copy()

    # Map/rename columns (dataset appears to have mismatched column descriptors)
    df['num_missing'] = pd.to_numeric(df['stdev_age'], errors='coerce')
    df['n_sockets'] = pd.to_numeric(df['prob_male'], errors='coerce')
    df['age_years'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df['sex_prob_male'] = pd.to_numeric(df['pop'], errors='coerce')

    # Original 'genus' column appears to be tooth class (Anterior/Posterior/Premolar)
    df['tooth_class'] = df['genus'].astype('category')

    # Original 'age' column appears to be specimen genus (Homo, Pan, Pongo, Papio)
    df['genus'] = df['age'].astype('category')

    # Create binary human indicator (case-insensitive contains 'homo')
    df['IsHuman'] = df['genus'].astype(str).str.lower().str.contains('homo').fillna(False).astype(int)

    # Drop rows with missing critical values
    df = df.dropna(subset=['num_missing', 'n_sockets', 'age_years', 'sex_prob_male', 'tooth_class', 'genus'])

    # Keep only rows with positive number of observable sockets
    df = df[df['n_sockets'] > 0]

    # Ensure numeric columns are finite
    df = df[np.isfinite(df['num_missing']) & np.isfinite(df['n_sockets']) & np.isfinite(df['age_years']) & np.isfinite(df['sex_prob_male'])]

    # Clip num_missing to valid range [0, n_sockets]
    df['num_missing'] = pd.to_numeric(df['num_missing'], errors='coerce').fillna(0)
    df['n_sockets'] = pd.to_numeric(df['n_sockets'], errors='coerce')
    df['num_missing'] = np.minimum(df['num_missing'], df['n_sockets'])
    df['num_missing'] = df['num_missing'].clip(lower=0)

    # Derived proportion for descriptive purposes; modeling will use binomial family with weights=n_sockets
    df['AMTL_rate'] = df['num_missing'] / df['n_sockets']

    # Make tooth_class explicitly categorical with sensible categories where possible
    df['tooth_class'] = df['tooth_class'].astype('category')
    df['genus'] = df['genus'].astype('category')

    # Final sanity-filter: require integer-like sockets and non-negative missing
    df = df[df['n_sockets'] >= 1]

    # Reset index for clean output
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial (logit) GLM to test whether modern humans have higher AMTL frequency
    than non-human primates, controlling for age, sex, and tooth class.

    Model specification (using proportions with weights = n_sockets):
      AMTL_rate ~ IsHuman + age_years + sex_prob_male + C(tooth_class)

    Weights: n_sockets (number of trials per observation)
    Family: Binomial (logit link)

    Returns the fitted GLM results object from statsmodels.
    """

    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    df_model = df.copy()

    # Ensure required columns are present
    required_cols = ['AMTL_rate', 'IsHuman', 'age_years', 'sex_prob_male', 'tooth_class', 'n_sockets']
    missing = [c for c in required_cols if c not in df_model.columns]
    if missing:
        raise ValueError('Dataframe is missing required columns for modeling: ' + ', '.join(missing))

    # Fit GLM: proportion with binomial family using weights = number of sockets
    formula = 'AMTL_rate ~ IsHuman + age_years + sex_prob_male + C(tooth_class)'
    glm_binom = smf.glm(formula=formula, data=df_model, family=sm.families.Binomial(), weights=df_model['n_sockets'])
    results = glm_binom.fit()

    # Print summary for immediate inspection (optional)
    print(results.summary())

    return results


