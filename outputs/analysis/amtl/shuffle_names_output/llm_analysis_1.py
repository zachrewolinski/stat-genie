from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/shuffle_names_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Gilmore (2013) AMTL dataset into a dataframe suitable for binomial regression.

    Mapping decisions (based on field descriptions in the provided schema):
    - 'age' column contains the specimen genus (Homo sapiens, Pan, Pongo, Papio) -> mapped to 'species'.
    - 'genus' column contains tooth-class labels (Anterior, Posterior, Premolar) -> mapped to 'tooth_class'.
    - 'prob_male' column contains counts of missing teeth for the tooth-class -> mapped to 'num_missing'.
    - 'sockets' column contains the number of observable sockets (trials) -> mapped to 'sockets'.
    - 'num_amtl' column contains estimated age at death -> mapped to 'age_at_death'.
    - 'stdev_age' contains assigned uncertainty of age at death -> mapped to 'age_sd'.
    - 'pop' contains an estimate/probability for sex (0-1) -> used to create SexMale.

    The transform will:
    - Coerce and clean the relevant columns.
    - Drop rows with missing/invalid trials (sockets <= 0) or missing successes.
    - Create integer counts for successes and trials where appropriate (rounding if floats present).
    - Create a binary is_human indicator and SexMale indicator.
    - Create AMTL_prop (proportion missing = num_missing / sockets) for modeling with binomial GLM using weights=sockets.
    """

    # Copy to avoid modifying original
    df = df.copy()

    # --- Map / rename columns into an internally consistent naming scheme ---
    # species (genus of specimen): column 'age' in provided data
    if 'age' in df.columns:
        df['species'] = df['age']
    else:
        df['species'] = df.get('species', pd.NA)

    # tooth class: column 'genus' in provided data
    if 'genus' in df.columns:
        df['tooth_class'] = df['genus']
    else:
        df['tooth_class'] = df.get('tooth_class', pd.NA)

    # num_missing: use 'prob_male' as the count of missing teeth (mapping from schema descriptions)
    if 'prob_male' in df.columns:
        df['num_missing'] = pd.to_numeric(df['prob_male'], errors='coerce')
    elif 'num_amtl' in df.columns and df['num_amtl'].dtype.kind in 'iu':
        df['num_missing'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    else:
        df['num_missing'] = pd.to_numeric(df.get('num_missing', pd.NA), errors='coerce')

    # sockets: number of observable sockets (trials)
    if 'sockets' in df.columns:
        df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    else:
        df['sockets'] = pd.to_numeric(df.get('sockets', pd.NA), errors='coerce')

    # age_at_death: from 'num_amtl' per schema description
    if 'num_amtl' in df.columns:
        df['age_at_death'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    else:
        df['age_at_death'] = pd.to_numeric(df.get('age_at_death', pd.NA), errors='coerce')

    # age uncertainty
    if 'stdev_age' in df.columns:
        df['age_sd'] = pd.to_numeric(df['stdev_age'], errors='coerce')
    else:
        df['age_sd'] = pd.to_numeric(df.get('age_sd', pd.NA), errors='coerce')

    # sex estimate/probability
    if 'pop' in df.columns:
        # pop described as estimate of sex in schema (0-1). Use >0.5 -> male
        df['sex_prob'] = pd.to_numeric(df['pop'], errors='coerce')
    else:
        df['sex_prob'] = pd.to_numeric(df.get('sex_prob', pd.NA), errors='coerce')

    # specimen id (for clustering later)
    if 'specimen' in df.columns:
        df['specimen'] = df['specimen'].astype(str)
    else:
        df['specimen'] = df.index.astype(str)

    # --- Clean counts: successes and trials must be non-negative integers ---
    # Round counts if they are floats aggregated across observations; then coerce to int
    df['num_missing'] = df['num_missing'].round().astype('Int64')
    df['sockets'] = df['sockets'].round().astype('Int64')

    # Drop rows where sockets is missing or not positive or num_missing is missing
    df = df.dropna(subset=['num_missing', 'sockets', 'species', 'tooth_class'])
    df = df[df['sockets'] > 0]

    # Ensure num_missing <= sockets and non-negative
    df = df[df['num_missing'] >= 0]
    df = df[df['num_missing'] <= df['sockets']]

    # Create a binary human indicator: True if species contains 'Homo' or equals 'Homo sapiens'
    df['species'] = df['species'].astype(str)
    df['is_human'] = df['species'].str.contains('Homo', case=False, na=False)

    # Create SexMale based on sex_prob if available; otherwise attempt to use a 'prob_male' style column
    if 'sex_prob' in df.columns and df['sex_prob'].notna().any():
        df['SexMale'] = (df['sex_prob'] > 0.5).astype(int)
    else:
        # fallback: try to detect a column called 'prob_male' that could be a probability
        if 'prob_male' in df.columns:
            # If values 0/1 or probabilities
            df['SexMale'] = (pd.to_numeric(df['prob_male'], errors='coerce') > 0.5).fillna(0).astype(int)
        else:
            # If no sex information available, set NA and later modeling will handle or drop
            df['SexMale'] = pd.NA

    # Proportion missing for GLM endog when using weights
    df['AMTL_prop'] = (df['num_missing'] / df['sockets']).astype(float)

    # Standardize tooth_class labels and species labels: strip whitespace
    df['tooth_class'] = df['tooth_class'].astype(str).str.strip().str.capitalize()
    df['species'] = df['species'].astype(str).str.strip()

    # Final drop of any rows that still have NA in essential columns for the model
    df = df.dropna(subset=['AMTL_prop', 'num_missing', 'sockets', 'is_human', 'tooth_class', 'age_at_death'])

    # Cast sockets and num_missing to int (regular python int) for statsmodels' weights/fit
    df['sockets'] = df['sockets'].astype(int)
    df['num_missing'] = df['num_missing'].astype(int)

    # Keep only the columns needed for modeling plus specimen id
    model_cols = [
        'specimen', 'species', 'is_human', 'tooth_class',
        'num_missing', 'sockets', 'AMTL_prop', 'age_at_death', 'age_sd', 'SexMale'
    ]
    for c in model_cols:
        if c not in df.columns:
            df[c] = pd.NA

    return df[model_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a binomial (logistic) GLM for AMTL frequency with clustering by specimen.

    Model specification:
    - Dependent variable: AMTL proportion (AMTL_prop) with weights = sockets (number of trials).
    - Key independent variable: is_human (binary; 1 = Homo sapiens, 0 = non-human primate).
    - Controls: age_at_death (continuous), SexMale (binary), tooth_class (categorical).
    - Clustered robust standard errors at the specimen level to account for within-specimen non-independence.

    Returns a dict with the fitted model and a clustered-robust covariance result object.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Ensure required columns exist
    required = ['AMTL_prop', 'sockets', 'is_human', 'age_at_death', 'SexMale', 'tooth_class', 'specimen', 'num_missing']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"The dataframe is missing required columns for modeling: {missing}")

    # Drop rows with NA in model variables
    mod_df = df.dropna(subset=['AMTL_prop', 'sockets', 'is_human', 'age_at_death', 'tooth_class'])

    # Convert categorical controls
    mod_df['tooth_class'] = mod_df['tooth_class'].astype('category')
    # is_human already boolean; coerce to int for modeling
    mod_df['is_human'] = mod_df['is_human'].astype(int)
    # SexMale: if NA (no sex information) we will impute as 0 and include an indicator if desired. Here drop NA to be conservative
    if mod_df['SexMale'].isna().any():
        mod_df = mod_df.dropna(subset=['SexMale'])
    mod_df['SexMale'] = mod_df['SexMale'].astype(int)

    # Formula: proportion outcome with weights = sockets. Use AMTL_prop as response.
    formula = 'AMTL_prop ~ is_human + age_at_death + SexMale + C(tooth_class)'

    # Fit GLM with binomial family using frequency weights = number of sockets (trials)
    glm_binom = smf.glm(formula=formula, data=mod_df,
                        family=sm.families.Binomial(),
                        freq_weights=mod_df['sockets'])
    res = glm_binom.fit()

    # Obtain clustered robust standard errors clustered by specimen
    # get_robustcov_results returns a results instance with adjusted covariances
    try:
        res_clustered = res.get_robustcov_results(cov_type='cluster', groups=mod_df['specimen'])
    except Exception as e:
        # If clustering fails, return the plain result and the exception message
        res_clustered = None
        cluster_error = str(e)
    else:
        cluster_error = None

    # Prepare a concise output
    output = {
        'model_result': res,
        'clustered_result': res_clustered,
        'cluster_error': cluster_error,
        'model_summary': res.summary().as_text()
    }

    return output


