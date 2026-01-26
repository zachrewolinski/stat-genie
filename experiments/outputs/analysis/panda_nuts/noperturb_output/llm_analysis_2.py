from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/panda_nuts/noperturb_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw nut-cracking dataset to produce the model-ready dataframe.

    Output columns required by the model:
      - chimpanzee (unchanged)
      - age (numeric)
      - sex_M (0/1 numeric: male = 1, female = 0)
      - help_Y (0/1 numeric: received help = 1, no = 0)
      - hammer (categorical, kept as-is)
      - nuts_opened (numeric, original)
      - seconds (numeric, original)
      - Efficiency (nuts per minute)
      - LogEfficiency (log1p(Efficiency)) -> dependent variable
    """

    # Make a copy to avoid modifying the original dataframe outside this function
    df = df.copy()

    # Standardize column names we expect; if not present this will raise KeyError for clarity
    expected_cols = ['chimpanzee', 'age', 'sex', 'hammer', 'nuts_opened', 'seconds', 'help']
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns: {missing}")

    # Drop rows with missing critical values
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age', 'sex', 'help', 'hammer', 'chimpanzee'])

    # Convert numeric columns to numeric types
    df['nuts_opened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Drop rows where seconds is non-positive or became NaN after conversion
    df = df[df['seconds'] > 0]
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age'])

    # Compute efficiency: nuts per minute
    df['Efficiency'] = (df['nuts_opened'] / df['seconds']) * 60.0

    # Log-transform efficiency using log1p to handle zero values: dependent variable
    df['LogEfficiency'] = np.log1p(df['Efficiency'])

    # Encode sex: male -> 1, female -> 0. Handle common capitalizations; unknown -> NaN
    df['sex_str'] = df['sex'].astype(str).str.lower().str.strip()
    df['sex_M'] = df['sex_str'].map({'m': 1, 'male': 1, 'f': 0, 'female': 0})

    # Encode help: any value starting with 'y' or 'Y' -> 1, otherwise 0 (handle 'N' and 'n')
    df['help_str'] = df['help'].astype(str).str.lower().str.strip()
    df['help_Y'] = df['help_str'].apply(lambda x: 1 if (isinstance(x, str) and x.startswith('y')) else 0)

    # Keep hammer as-is (categorical). Ensure it's a string type for modeling
    df['hammer'] = df['hammer'].astype(str)

    # Ensure chimpanzee id is present and appropriate type
    # Keep original values (int or str) as grouping factor for mixed models
    df['chimpanzee'] = df['chimpanzee']

    # Drop any rows where encoding produced NaNs (e.g., unknown sex)
    df = df.dropna(subset=['sex_M'])

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    # Keep only columns required for modeling + useful diagnostics
    keep_cols = ['chimpanzee', 'age', 'sex_M', 'help_Y', 'hammer', 'nuts_opened', 'seconds', 'Efficiency', 'LogEfficiency']
    df = df[keep_cols]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a linear mixed-effects model predicting LogEfficiency from age, sex, help, and hammer,
    with a random intercept for chimpanzee to account for repeated measures.

    Model formula (fixed effects):
      LogEfficiency ~ age + sex_M + help_Y + age:help_Y + sex_M:help_Y + C(hammer)

    Random effect:
      random intercept for chimpanzee (groups=df['chimpanzee'])

    Returns the fitted MixedLMResults object. If the mixed model fails to converge, falls back to OLS.
    """

    import statsmodels.formula.api as smf
    import warnings

    # Copy to avoid side-effects
    data = df.copy()

    # Verify required columns
    required = ['LogEfficiency', 'age', 'sex_M', 'help_Y', 'hammer', 'chimpanzee']
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise KeyError(f"Missing required columns for modeling: {missing}")

    # Ensure categorical hammer is treated as category
    data['hammer'] = data['hammer'].astype('category')

    # Define formula
    formula = 'LogEfficiency ~ age + sex_M + help_Y + age:help_Y + sex_M:help_Y + C(hammer)'

    # Fit mixed effects model with random intercept for chimpanzee
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            md = smf.mixedlm(formula, data=data, groups=data['chimpanzee'], re_formula='1')
            mdf = md.fit(reml=False)
        # Return the fitted model object
        return mdf
    except Exception as e:
        # Fallback: OLS using the same fixed-effects formula (no random intercept)
        try:
            ols_md = smf.ols(formula, data=data).fit()
            # Attach the exception info so the user knows a fallback occurred
            ols_md._fallback_reason = str(e)
            return ols_md
        except Exception as e2:
            raise RuntimeError(f"Both MixedLM and OLS failed. MixedLM error: {e}; OLS error: {e2}")


