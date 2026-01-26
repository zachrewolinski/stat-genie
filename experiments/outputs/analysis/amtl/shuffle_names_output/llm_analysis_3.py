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
    Transform the raw dataframe into the analytic dataframe with the exact columns used by the models.

    Assumptions / mapping based on provided schema (columns in raw df):
    - 'stdev_age' contains the observed number of missing teeth for the relevant tooth class (AMTL count). We round to integer.
    - 'prob_male' contains the number of observable sockets (n_sockets). We round to integer.
    - 'num_amtl' contains the estimated age at death (years) for specimen (numeric).
    - 'pop' contains the sex estimate/probability of male (numeric 0-1).
    - 'age' contains the specimen genus (e.g., 'Homo sapiens', 'Pan', 'Pongo', 'Papio').
    - 'genus' contains the tooth class label (e.g., 'Anterior', 'Posterior', 'Premolar').

    The function produces these final columns (exact names used in model):
      - AMTL: integer count of missing teeth
      - n_sockets: integer count of observable sockets
      - AMTL_prop: AMTL / n_sockets (float)
      - Genus: categorical genus string
      - IsHuman: binary indicator (1 if Genus contains 'Homo')
      - Age: numeric age at death (years)
      - Age_z: standardized age (mean 0, sd 1)
      - SexProb: probability of being male (0-1)
      - ToothClass: tooth class categorical

    The function also filters out invalid rows (missing critical values, n_sockets <= 0, or AMTL > n_sockets).
    """

    df = df.copy()

    # Convert/clean counts and measures. Errors -> NaN
    df['AMTL'] = pd.to_numeric(df.get('stdev_age'), errors='coerce').round()
    df['n_sockets'] = pd.to_numeric(df.get('prob_male'), errors='coerce').round()
    df['Age'] = pd.to_numeric(df.get('num_amtl'), errors='coerce')
    df['SexProb'] = pd.to_numeric(df.get('pop'), errors='coerce')

    # Genus and tooth class as strings
    df['Genus'] = df.get('age').astype(str)
    df['ToothClass'] = df.get('genus').astype(str)

    # Drop rows with missing critical variables
    df = df.dropna(subset=['AMTL', 'n_sockets', 'Age', 'SexProb', 'Genus', 'ToothClass'])

    # Convert AMTL and n_sockets to integer dtype after dropna
    df['AMTL'] = df['AMTL'].astype(int)
    df['n_sockets'] = df['n_sockets'].astype(int)

    # Keep only rows with at least one observable socket and sensible counts
    df = df[df['n_sockets'] > 0]
    df = df[df['AMTL'] <= df['n_sockets']]
    df = df[df['AMTL'] >= 0]

    # Compute proportion
    df['AMTL_prop'] = df['AMTL'] / df['n_sockets']

    # Binary human indicator (Homo sapiens)
    df['IsHuman'] = df['Genus'].str.contains('Homo', case=False, na=False).astype(int)

    # Standardize Age for easier model interpretation
    df['Age_z'] = (df['Age'] - df['Age'].mean()) / (df['Age'].std(ddof=0) if df['Age'].std(ddof=0) != 0 else 1)

    # Keep and return only columns needed for modeling (plus specimen id if present)
    cols_to_keep = []
    if 'specimen' in df.columns:
        cols_to_keep.append('specimen')
    cols_to_keep += ['AMTL', 'n_sockets', 'AMTL_prop', 'Genus', 'IsHuman', 'Age', 'Age_z', 'SexProb', 'ToothClass']

    return df[cols_to_keep]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit binomial regression models to test whether modern humans have higher AMTL rates
    than non-human primates, controlling for age, sex, and tooth class.

    Two complementary models are returned:
      - model_is_human: GLM binomial with IsHuman as the primary predictor (binary human vs non-human).
      - model_by_genus: GLM binomial with Genus (categorical) to compare across genera.

    Both models use AMTL_prop as the response and n_sockets as binomial weights.

    Returns a dictionary with fitted statsmodels results objects.
    """

    df = df.copy()

    # Ensure the needed columns exist
    required = ['AMTL_prop', 'n_sockets', 'IsHuman', 'Age_z', 'SexProb', 'ToothClass', 'Genus']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Create tooth-class dummy variables (drop first to avoid collinearity)
    tooth_dummies = pd.get_dummies(df['ToothClass'].astype(str), prefix='Tooth', drop_first=True)

    # Model 1: binary human vs non-human
    exog1 = pd.concat([df[['IsHuman', 'Age_z', 'SexProb']].reset_index(drop=True), tooth_dummies.reset_index(drop=True)], axis=1)
    exog1 = sm.add_constant(exog1, has_constant='add')

    glm1 = sm.GLM(df['AMTL_prop'], exog1, family=sm.families.Binomial(), var_weights=df['n_sockets'])
    res1 = glm1.fit()

    # Model 2: genus-level categorical comparison
    genus_dummies = pd.get_dummies(df['Genus'].astype(str), prefix='Genus', drop_first=True)
    exog2 = pd.concat([genus_dummies.reset_index(drop=True), df[['Age_z', 'SexProb']].reset_index(drop=True), tooth_dummies.reset_index(drop=True)], axis=1)
    exog2 = sm.add_constant(exog2, has_constant='add')

    glm2 = sm.GLM(df['AMTL_prop'], exog2, family=sm.families.Binomial(), var_weights=df['n_sockets'])
    res2 = glm2.fit()

    # Return both fitted results so the caller can inspect coefficients, CIs, p-values, diagnostics
    return {
        'model_is_human': res1,
        'model_by_genus': res2
    }


