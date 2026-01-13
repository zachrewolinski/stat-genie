from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import patsy

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/add_features_output/amtl.csv')


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataframe for binomial regression of AMTL.
    Produces columns required by the model:
      - num_amtl: integer count of missing teeth (as provided)
      - sockets: integer count of observable sockets (trials)
      - IsHuman: binary indicator (1 if genus == 'Homo sapiens', else 0)
      - age_c: age centered on the sample median
      - prob_male: numeric between 0 and 1 (as provided)
      - tooth_class: categorical with expected values 'Anterior','Premolar','Posterior'

    This function also filters invalid or missing rows that would make binomial modeling invalid
    (e.g., sockets <= 0 or num_amtl outside [0, sockets]).
    """
    df = df.copy()

    # Ensure necessary columns exist
    required_cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    # Convert numeric columns and drop rows with NA in core columns
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')

    df = df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class'])

    # Remove impossible/invalid rows
    df = df[df['sockets'] > 0]
    df = df[(df['num_amtl'] >= 0) & (df['num_amtl'] <= df['sockets'])]

    # Convert counts to integer type (safe now after filtering)
    df['num_amtl'] = df['num_amtl'].astype(int)
    df['sockets'] = df['sockets'].astype(int)

    # Create binary human indicator
    # Trim whitespace and standardize genus strings before comparison
    df['genus'] = df['genus'].astype(str).str.strip()
    df['IsHuman'] = (df['genus'] == 'Homo sapiens').astype(int)

    # Center age at median (robust to skew)
    median_age = df['age'].median()
    df['age_c'] = df['age'] - median_age

    # Ensure prob_male is between 0 and 1; remove rows outside that range as likely data errors
    df = df[(df['prob_male'] >= 0.0) & (df['prob_male'] <= 1.0)]

    # Clean tooth_class and make it categorical with expected levels
    df['tooth_class'] = df['tooth_class'].astype(str).str.strip()
    allowed = ['Anterior', 'Premolar', 'Posterior']
    df = df[df['tooth_class'].isin(allowed)]
    df['tooth_class'] = pd.Categorical(df['tooth_class'], categories=allowed)

    # Final check: at least some variation in IsHuman
    if df['IsHuman'].nunique() < 2:
        raise ValueError('No variation in IsHuman after filtering; cannot estimate effect.')

    # Keep only the columns required for modeling to make the output clear
    keep_cols = ['num_amtl', 'sockets', 'IsHuman', 'age_c', 'prob_male', 'tooth_class']
    df = df[keep_cols]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial GLM to test whether modern humans have higher AMTL than non-human primates
    while controlling for age, sex (prob_male), and tooth class.

    Model specification (using counts):
      response = (num_amtl, sockets - num_amtl)  (successes, failures)
      predictors = IsHuman + age_c + prob_male + C(tooth_class)

    Returns the fitted GLMResults object (statsmodels GeneralizedLinearResults).
    """
    # Validate required columns are present
    required = ['num_amtl', 'sockets', 'IsHuman', 'age_c', 'prob_male', 'tooth_class']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe is missing required columns for modeling: {missing}")

    # Build response as a 2-column array: [successes, failures]
    successes = np.asarray(df['num_amtl'], dtype=np.float64)
    failures = np.asarray(df['sockets'] - df['num_amtl'], dtype=np.float64)
    endog = np.vstack([successes, failures]).T  # shape (nobs, 2)

    # Build design matrix (including intercept) using patsy to ensure proper handling of categorical tooth_class
    exog = patsy.dmatrix('IsHuman + age_c + prob_male + C(tooth_class)', df, return_type='dataframe')

    # Fit GLM with Binomial family using count response form (endog with shape (n,2))
    glm_model = sm.GLM(endog, exog, family=sm.families.Binomial())
    results = glm_model.fit()

    return results