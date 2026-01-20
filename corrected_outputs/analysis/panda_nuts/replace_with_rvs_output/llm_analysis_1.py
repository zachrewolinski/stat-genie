from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/panda_nuts/replace_with_rvs_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the dataframe used for modeling.

    Produces these new/clean columns used in the model:
    - NutsPerMin: nuts_opened per minute (nuts_opened * 60 / seconds)
    - Sex_Male: binary (1 = 'm', 0 = 'f')
    - Help_Received: binary (1 = 'y', 0 = 'n')
    - hammer: categorical (kept as-is but cast to category dtype)
    - chimpanzee: categorical (grouping variable)

    Drops rows with missing/invalid numeric data required to compute efficiency.
    """
    # Work on a copy
    df = df.copy()

    # Ensure numeric columns are numeric
    df['nuts_opened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Drop rows missing core outcome or duration information
    df = df.dropna(subset=['nuts_opened', 'seconds'])

    # Remove invalid session durations (zero or negative)
    df = df[df['seconds'] > 0]

    # Compute efficiency: nuts opened per minute
    df['NutsPerMin'] = df['nuts_opened'] * 60.0 / df['seconds']

    # Encode sex into binary variable Sex_Male (1 = male, 0 = female). If unknown, results become NaN and are dropped below.
    df['Sex_Male'] = df['sex'].astype(str).str.lower().map({'m': 1, 'f': 0})

    # Encode help into binary variable Help_Received (1 = yes, 0 = no). Map common case-insensitive values.
    df['Help_Received'] = df['help'].astype(str).str.lower().map({'y': 1, 'n': 0})

    # Cast hammer and chimpanzee to categorical for modeling
    df['hammer'] = df['hammer'].astype('category')
    # Keep chimpanzee as category (grouping variable)
    df['chimpanzee'] = df['chimpanzee'].astype('category')

    # Drop rows where the derived IVs are missing (unmappable sex/help values) or where NutsPerMin is missing
    df = df.dropna(subset=['NutsPerMin', 'Sex_Male', 'Help_Received'])

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a linear mixed-effects model predicting nut-cracking efficiency (NutsPerMin)
    from age, sex, and help, controlling for hammer type and accounting for repeated
    measurements within chimpanzees via a random intercept.

    Model formula:
      NutsPerMin ~ age + Sex_Male + Help_Received + C(hammer)
    Random effects:
      random intercept for chimpanzee (groups=df['chimpanzee'])

    Returns the fitted mixed-effects model results object.
    """
    import statsmodels.formula.api as smf

    # Ensure required columns exist
    required_cols = ['NutsPerMin', 'age', 'Sex_Male', 'Help_Received', 'hammer', 'chimpanzee']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataframe is missing required columns for modeling: {missing}")

    # Fit mixed-effects model with random intercept for chimpanzee
    formula = "NutsPerMin ~ age + Sex_Male + Help_Received + C(hammer)"
    md = smf.mixedlm(formula, data=df, groups=df['chimpanzee'])

    # Fit the model (use REML=False for likelihood-based comparisons; change if you prefer REML)
    mdf = md.fit(reml=False)

    # Print summary for quick inspection; return the fitted model object
    print(mdf.summary())
    return mdf


