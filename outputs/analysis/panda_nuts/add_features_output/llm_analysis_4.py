from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/panda_nuts/add_features_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe suitable for modeling.

    Produces the following columns required by the model:
      - nuts_per_min: nuts opened per minute (continuous DV)
      - age_z: age standardized (IV)
      - Sex_M: sex coded 1=male, 0=female (IV)
      - HelpReceived: help coded 1=yes, 0=no (IV)
      - hammer: hammer type as categorical (control)
      - chimp_id: individual ID for random effects (control)

    Steps:
      - Drop rows missing required fields or with invalid session duration (seconds <= 0)
      - Compute nuts_per_min = nuts_opened / seconds * 60
      - Standardize age (z-score)
      - Recode sex and help to binary columns
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Required columns
    required_cols = ['chimpanzee', 'age', 'sex', 'nuts_opened', 'seconds', 'help', 'hammer']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Drop rows with NA in required fields
    df = df.dropna(subset=['chimpanzee', 'age', 'sex', 'nuts_opened', 'seconds', 'help'])

    # Remove rows with non-positive seconds (avoid divide-by-zero)
    df = df[df['seconds'].astype(float) > 0]

    # Convert numeric columns
    df['nuts_opened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # After coercion drop rows where conversion failed
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age'])

    # Dependent variable: nuts per minute
    df['nuts_per_min'] = df['nuts_opened'] / df['seconds'] * 60.0

    # Independent: age standardized (z-score)
    age_mean = df['age'].mean()
    age_std = df['age'].std(ddof=0) if df['age'].std(ddof=0) != 0 else 1.0
    df['age_z'] = (df['age'] - age_mean) / age_std

    # Recode sex: expect values like 'm'/'f' or uppercase variants
    df['sex_str'] = df['sex'].astype(str).str.strip().str.lower()
    df['Sex_M'] = df['sex_str'].map(lambda x: 1 if x == 'm' or x == 'male' else (0 if x == 'f' or x == 'female' else np.nan))

    # Recode help: expect 'y' and 'N' per schema (case-insensitive)
    df['help_str'] = df['help'].astype(str).str.strip().str.lower()
    df['HelpReceived'] = df['help_str'].map(lambda x: 1 if x in ['y', 'yes', 'yep', 'true'] else (0 if x in ['n', 'no', 'false'] else np.nan))

    # Convert chimpanzee ID to a consistent column name for grouping
    df['chimp_id'] = df['chimpanzee']

    # Hammer type keep as categorical control (clean strings)
    df['hammer'] = df['hammer'].astype(str).str.strip()

    # Drop rows where recoding failed (unknown sex or help values)
    df = df.dropna(subset=['Sex_M', 'HelpReceived'])

    # Optionally, convert to appropriate dtypes
    df['Sex_M'] = df['Sex_M'].astype(int)
    df['HelpReceived'] = df['HelpReceived'].astype(int)

    # Final sanity: drop rows with NaN DV
    df = df.dropna(subset=['nuts_per_min'])

    # Reset index
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a linear mixed-effects model predicting nut-cracking efficiency (nuts_per_min)
    from age, sex, and help, controlling for hammer type and including a random
    intercept for chimpanzee ID to account for repeated measures.

    Model formula:
      nuts_per_min ~ age_z + Sex_M + HelpReceived + Sex_M:HelpReceived + age_z:HelpReceived + C(hammer)

    Random effects:
      random intercept for chimp_id

    Returns the fitted statsmodels result object.
    """
    import statsmodels.formula.api as smf

    # Ensure required columns exist
    required = ['nuts_per_min', 'age_z', 'Sex_M', 'HelpReceived', 'hammer', 'chimp_id']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Fit mixed effects model (random intercept by chimp_id)
    formula = 'nuts_per_min ~ age_z + Sex_M + HelpReceived + Sex_M:HelpReceived + age_z:HelpReceived + C(hammer)'
    md = smf.mixedlm(formula, df, groups=df['chimp_id'], re_formula='1')

    # Use maximum likelihood (reml=False) for easier comparison if needed
    try:
        mdf = md.fit(reml=False, method='lbfgs')
    except Exception:
        # fallback to default optimizer/REML if lbfgs fails
        mdf = md.fit()

    # Print a brief summary and return the fitted model object
    print(mdf.summary())
    return mdf


