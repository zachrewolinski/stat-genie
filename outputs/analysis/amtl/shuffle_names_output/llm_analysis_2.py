from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Gilmore (2013) AMTL dataset into a dataframe ready for binomial modeling.

    Output columns required by the model:
      - amtl_count : integer count of missing teeth (successes)
      - n_sockets  : integer count of observable sockets (trials)
      - Species    : taxonomic group / species (e.g., 'Homo sapiens', 'Pan', 'Pongo', 'Papio')
      - ToothClass : tooth class (e.g., 'Anterior', 'Premolar', 'Posterior')
      - AgeAtDeath : estimated age at death (continuous)
      - AgeScaled  : standardized age (zero mean, unit sd)
      - Male       : binary sex indicator (1 = male, 0 = female)

    Notes on column mapping (based on dataset schema):
      - 'stdev_age' is treated as the count of missing teeth for that specimen & tooth class (amtl_count).
      - 'sockets' is treated as the number of observable tooth sockets (n_sockets, trials).
      - 'num_amtl' is treated as the estimated age at death (AgeAtDeath).
      - 'age' column contains the taxonomic group (genus/species) and is renamed to 'Species'.
      - 'genus' column in the raw data contains tooth class labels (Anterior, Posterior, Premolar) and is renamed to 'ToothClass'.
      - 'pop' is treated as an estimate/probability of maleness (0-1); Male = (pop >= 0.5).

    The function is robust to small inconsistencies: it rounds counts, ensures integer sockets > 0, and clips amtl_count to [0, n_sockets]. Rows missing critical information are dropped.
    """
    df = df.copy()

    # --- Ensure necessary columns exist; if not, raise an informative error ---
    required = ['stdev_age', 'sockets', 'age', 'genus', 'num_amtl']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe is missing required column(s): {missing}")

    # --- Derive amtl_count and n_sockets as integer counts ---
    df['amtl_count'] = pd.to_numeric(df['stdev_age'], errors='coerce').round().astype('Int64')
    df['n_sockets'] = pd.to_numeric(df['sockets'], errors='coerce').round().astype('Int64')

    # Drop rows where we cannot interpret sockets or amtl_count
    df = df[~df['n_sockets'].isna()]
    df = df[~df['amtl_count'].isna()]

    # Convert to plain ints (pandas nullable int -> normal int) after dropping NA
    df['n_sockets'] = df['n_sockets'].astype(int)
    df['amtl_count'] = df['amtl_count'].astype(int)

    # Drop rows with non-positive trial counts (must have at least one socket to be informative)
    df = df[df['n_sockets'] > 0]

    # Clip amtl_count so 0 <= amtl_count <= n_sockets
    df['amtl_count'] = df.apply(lambda r: max(0, min(r['amtl_count'], r['n_sockets'])), axis=1)

    # --- Species and ToothClass ---
    # 'age' column in schema contains genus/species labels
    df['Species'] = df['age'].astype(str).str.strip()
    # 'genus' column in schema contains tooth class labels (Anterior/Posterior/Premolar)
    df['ToothClass'] = df['genus'].astype(str).str.strip()

    # Normalize ToothClass labels (case-insensitive mapping) and drop weird entries
    df['ToothClass'] = df['ToothClass'].str.lower().replace({
        'anterior': 'Anterior',
        'posterior': 'Posterior',
        'premolar': 'Premolar'
    })

    # Keep only canonical tooth classes (safety filter)
    df = df[df['ToothClass'].isin(['Anterior', 'Posterior', 'Premolar'])]

    # --- Age at death ---
    df['AgeAtDeath'] = pd.to_numeric(df['num_amtl'], errors='coerce')

    # Drop rows with missing AgeAtDeath, since age is an important control
    df = df[~df['AgeAtDeath'].isna()]

    # Standardize age for modeling (use population std dev, ddof=0)
    age_std = df['AgeAtDeath'].std(ddof=0)
    if pd.isna(age_std) or age_std == 0:
        age_std = 1.0
    df['AgeScaled'] = (df['AgeAtDeath'] - df['AgeAtDeath'].mean()) / age_std

    # --- Sex / Male indicator ---
    if 'pop' in df.columns:
        df['Male'] = pd.to_numeric(df['pop'], errors='coerce').fillna(0)
        df['Male'] = (df['Male'] >= 0.5).astype(int)
    else:
        if 'prob_male' in df.columns:
            pm = pd.to_numeric(df['prob_male'], errors='coerce')
            if pm.between(0, 1).all():
                df['Male'] = (pm >= 0.5).astype(int)
            else:
                df['Male'] = 0
        else:
            df['Male'] = 0

    # --- Final selection of columns required for modeling ---
    out_cols = ['amtl_count', 'n_sockets', 'Species', 'ToothClass', 'AgeAtDeath', 'AgeScaled', 'Male']
    df_out = df[out_cols].reset_index(drop=True)

    return df_out


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial (logistic) regression (GLM) modeling AMTL frequency while accounting for species,
    age at death, sex, and tooth class. The model uses the count of missing teeth (amtl_count) out of
    the number of observable sockets (n_sockets) as the binomial response.

    Returns the fitted GLMResults object from statsmodels, or None if there is no data to fit.
    """
    # Work on a copy
    df = df.copy()

    # Basic sanity checks
    required_cols = {'amtl_count', 'n_sockets', 'Species', 'ToothClass', 'AgeScaled', 'Male'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f'Dataframe must contain amtl_count, n_sockets, Species, ToothClass, AgeScaled, and Male columns. Missing: {missing}')

    # Remove rows with invalid socket counts
    df = df[df['n_sockets'] > 0].copy()

    # If no data left, return None (no fit possible) rather than raising an exception
    if df.shape[0] == 0:
        return None

    # Compute observed proportion (endog for GLM with weights)
    df['prop_amtl'] = df['amtl_count'] / df['n_sockets']

    # Build design matrix manually to avoid patsy/formula issues when categorical variables have limited levels.
    # Use one-hot encoding for Species and ToothClass, dropping the first level to avoid multicollinearity.
    species_dummies = pd.get_dummies(df['Species'].astype(str), prefix='Species', drop_first=True)
    tooth_dummies = pd.get_dummies(df['ToothClass'].astype(str), prefix='ToothClass', drop_first=True)

    exog_parts = [df[['AgeScaled', 'Male']].reset_index(drop=True)]
    if not species_dummies.empty:
        exog_parts.append(species_dummies.reset_index(drop=True))
    if not tooth_dummies.empty:
        exog_parts.append(tooth_dummies.reset_index(drop=True))

    exog = pd.concat(exog_parts, axis=1)
    # Add intercept
    exog = sm.add_constant(exog, has_constant='add')

    # Ensure exog is numeric
    exog = exog.astype(float)

    # Endogenous variable as proportion, with frequency weights equal to n_sockets
    endog = df['prop_amtl'].astype(float)

    # Fit GLM with binomial family using frequency weights (number of trials)
    model_glm = sm.GLM(endog, exog, family=sm.families.Binomial())
    try:
        results = model_glm.fit(freq_weights=df['n_sockets'])
    except TypeError:
        # Fallback to weights keyword if freq_weights unsupported
        results = model_glm.fit(weights=df['n_sockets'])

    return results