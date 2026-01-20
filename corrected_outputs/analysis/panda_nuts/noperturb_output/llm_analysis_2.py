from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/panda_nuts/noperturb_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw nut-cracking dataset to produce the columns required for modeling.

    Produces:
      - Efficiency: nuts opened per second (float)
      - log_efficiency: np.log1p(Efficiency) (float) used as the DV
      - Sex_M: binary sex indicator (1 = male, 0 = female)
      - Help_Y: binary help indicator (1 = received help, 0 = no help)
      - Ensures 'chimpanzee' and 'hammer' are present and appropriate dtypes
    """
    # Work on a copy
    df = df.copy()

    # Standardize column names lower/strip if necessary (we assume provided names are exact in schema)
    # Drop rows with missing key variables
    required = ['chimpanzee', 'age', 'sex', 'help', 'nuts_opened', 'seconds', 'hammer']
    df = df.dropna(subset=required)

    # Ensure numeric fields have proper numeric dtype
    df['nuts_opened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Remove rows with non-positive session duration or missing numeric values
    df = df[df['seconds'] > 0]
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age'])

    # Compute raw efficiency (nuts per second)
    df['Efficiency'] = df['nuts_opened'] / df['seconds']

    # Log-transform the efficiency to stabilize variance and handle zeros: log1p(Efficiency)
    df['log_efficiency'] = np.log1p(df['Efficiency'])

    # Encode sex into binary indicator Sex_M: 1 if male, 0 if female
    # handle various capitalization
    df['Sex_M'] = (df['sex'].astype(str).str.lower().map({'m': 1, 'male': 1, 'f': 0, 'female': 0}))

    # Encode help into binary indicator Help_Y: 1 if received help, 0 if not
    df['Help_Y'] = (df['help'].astype(str).str.lower().map({'y': 1, 'yes': 1, 'n': 0, 'no': 0}))

    # Drop rows where mappings failed
    df = df.dropna(subset=['Sex_M', 'Help_Y'])

    # Ensure chimpanzee ID is treated as categorical/grouping variable (string)
    df['chimpanzee'] = df['chimpanzee'].astype(str)

    # Ensure hammer is categorical and drop rows with missing/empty hammer
    df['hammer'] = df['hammer'].astype(str)
    df = df.dropna(subset=['hammer'])

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit a mixed-effects linear model predicting log-transformed efficiency from age, sex, and help,
    controlling for hammer type and including a random intercept for chimpanzee.

    Model specification:
      log_efficiency ~ age + Sex_M + Help_Y + age:Help_Y + C(hammer)
      random intercept by chimpanzee

    Returns the fitted model results object. If the mixed-effects model fails to converge or errors,
    falls back to an OLS fit with the same fixed-effects formula (no random effects).
    """
    import statsmodels.api as sm

    # Verify required columns are present
    required_cols = ['log_efficiency', 'age', 'Sex_M', 'Help_Y', 'hammer', 'chimpanzee']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Try mixed-effects model with random intercept for chimpanzee
    formula = 'log_efficiency ~ age + Sex_M + Help_Y + age:Help_Y + C(hammer)'

    try:
        md = sm.MixedLM.from_formula(formula, groups='chimpanzee', data=df)
        mdf = md.fit(reml=False)
        return mdf
    except Exception as e:
        # Fallback: ordinary least squares (no random effects)
        import statsmodels.formula.api as smf
        ols = smf.ols(formula, data=df).fit()
        return ols


