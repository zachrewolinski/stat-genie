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
    Transform the original dataframe into a clean dataframe for binomial modeling of AMTL.

    Notes on column mapping: the provided schema had inconsistent descriptions. The code below uses the original column names but maps them to explicit, interpretable columns used in the model:
      - 'stdev_age' contains integer counts corresponding to number of missing teeth for the given tooth class -> mapped to n_missing
      - 'prob_male' contains integer counts corresponding to number of observable sockets for the given tooth class -> mapped to n_sockets
      - 'num_amtl' contains estimated age at death (years) -> mapped to age_at_death
      - 'pop' contains a numeric 0-1 estimate interpretable as probability male -> mapped to sex_prob_male
      - 'age' contains genus names (Homo sapiens, Pan, Pongo, Papio) -> mapped to Genus
      - 'genus' contains tooth class labels (Anterior, Posterior, Premolar) -> mapped to ToothClass

    The function creates clear column names and computes derived variables used in modeling.
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Map ambiguous columns into clear variables used in the model
    # Based on the dataset schema/values, create standardized columns:
    # n_missing: integer count of AMTL for that specimen/tooth-class (from 'stdev_age')
    # n_sockets: integer number of observable sockets (from 'prob_male')
    # age_at_death: estimated age (from 'num_amtl')
    # sex_prob_male: numeric 0-1 probability estimate of maleness (from 'pop')
    # Genus: specimen genus (from 'age')
    # ToothClass: tooth class (from 'genus')

    # Rename/make new columns (do not drop originals)
    df['n_missing'] = pd.to_numeric(df['stdev_age'], errors='coerce')
    df['n_sockets'] = pd.to_numeric(df['prob_male'], errors='coerce')
    df['age_at_death'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df['sex_prob_male'] = pd.to_numeric(df['pop'], errors='coerce')

    # Genus (Homo/ Pan / Pongo / Papio) - present in column named 'age' according to schema
    df['Genus'] = df['age'].astype(str)

    # ToothClass (Anterior/Posterior/Premolar) - present in column named 'genus' according to schema
    df['ToothClass'] = df['genus'].astype(str)

    # Drop rows that are missing the essential count data
    df = df.dropna(subset=['n_missing', 'n_sockets'])

    # Ensure integer counts for missing and sockets
    # Round n_missing and n_sockets to nearest integer, but keep rows where sockets > 0 and 0 <= n_missing <= n_sockets
    df['n_sockets'] = df['n_sockets'].round().astype(int)
    df['n_missing'] = df['n_missing'].round().astype(int)

    # Remove impossible rows
    df = df[df['n_sockets'] > 0]
    df = df[(df['n_missing'] >= 0) & (df['n_missing'] <= df['n_sockets'])]

    # Age: if available, coerce to float; missing ages remain NaN but we will keep rows (model can handle or drop later)
    df['age_at_death'] = pd.to_numeric(df['age_at_death'], errors='coerce')

    # Sex: derive Male binary from sex_prob_male when available (threshold 0.5). Keep sex_prob_male column too.
    df['sex_prob_male'] = pd.to_numeric(df['sex_prob_male'], errors='coerce')
    df['Male'] = df['sex_prob_male'].apply(lambda x: 1 if pd.notnull(x) and x >= 0.5 else (0 if pd.notnull(x) else np.nan))

    # Create binary IsHuman indicator (primary IV)
    df['IsHuman'] = df['Genus'].str.strip().apply(lambda x: 1 if x == 'Homo sapiens' or x == 'Homo' or x == 'Homo_sapiens' else 0)

    # Create AMTL_rate for descriptive checks (proportion)
    df['AMTL_rate'] = df['n_missing'] / df['n_sockets']

    # Normalize ToothClass categories (strip whitespace)
    df['ToothClass'] = df['ToothClass'].str.strip().replace({'Anterior': 'Anterior', 'Posterior': 'Posterior', 'Premolar': 'Premolar'})

    # Optionally drop rows with missing key covariates. Keep rows with missing sex or age but the model call can dropna if required.
    # For reproducible modeling, drop rows with missing Genus or ToothClass (these are required categorical variables)
    df = df.dropna(subset=['Genus', 'ToothClass'])

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a binomial GLM predicting AMTL (n_missing out of n_sockets) from IsHuman,
    controlling for age_at_death, Male, and ToothClass.

    Returns the fitted GLMResults object from statsmodels.
    """
    import statsmodels.api as sm
    import numpy as np
    import pandas as pd

    # Work on a copy
    data = df.copy()

    # Drop rows missing variables required for the model
    # We require n_missing, n_sockets, IsHuman, ToothClass. Age and Male are included as controls; drop rows with missing age or male only if you prefer complete-case analysis.
    data = data.dropna(subset=['n_missing', 'n_sockets', 'IsHuman', 'ToothClass'])

    # Prepare endog as a two-column array of [successes, failures] for binomial
    successes = data['n_missing'].astype(int).values
    failures = (data['n_sockets'].astype(int) - data['n_missing'].astype(int)).values
    endog = np.vstack([successes, failures]).T

    # Build design matrix (exog): intercept + IsHuman + age_at_death + Male + ToothClass dummies (drop one level)
    exog_vars = []
    # Always include IsHuman
    exog = pd.DataFrame({'IsHuman': data['IsHuman'].astype(float)})

    # Include age_at_death (centered) if available
    if 'age_at_death' in data.columns:
        # center age to aid interpretation
        age_mean = data['age_at_death'].mean()
        exog['age_at_death'] = (data['age_at_death'] - age_mean).fillna(0.0)
    else:
        exog['age_at_death'] = 0.0

    # Include Male (if missing values remain, fill with 0.5 to avoid dropping rows; alternatively dropna before)
    if 'Male' in data.columns:
        exog['Male'] = data['Male'].fillna(0.5).astype(float)
    else:
        exog['Male'] = 0.0

    # ToothClass categorical dummies (drop first to avoid multicollinearity)
    tooth_dummies = pd.get_dummies(data['ToothClass'].astype(str), prefix='Tooth', drop_first=True)
    # If there are no tooth dummies (i.e., only one class), this results in empty DataFrame — handle gracefully
    if tooth_dummies.shape[1] > 0:
        exog = pd.concat([exog, tooth_dummies.reset_index(drop=True)], axis=1)

    # Add intercept
    exog = sm.add_constant(exog, has_constant='add')

    # Fit binomial GLM using the (successes, failures) representation
    model = sm.GLM(endog, exog, family=sm.families.Binomial())
    results = model.fit()

    # For convenience, attach the data used and the design matrix to the results object (nonstandard but helpful)
    results.model_data = {
        'endog_successes': successes,
        'endog_failures': failures,
        'exog': exog,
        'data_rows': data.index.tolist()
    }

    return results


