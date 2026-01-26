from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/panda_nuts/negative_leading_statement_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw nut-cracking dataset to produce the variables used in modeling.

    Produces the following new columns used by the model:
      - Sex_M: binary (1 = male, 0 = female)
      - Help: binary (1 = helped, 0 = not helped)
      - Efficiency: nuts_opened / seconds (nuts per second)
      - LogEfficiency: natural log(Efficiency + eps)
    Also enforces types for hammer and chimpanzee (categorical) and drops invalid rows.
    """
    df = df.copy()

    # Ensure required columns exist and drop rows with missing critical values
    req_cols = ['nuts_opened', 'seconds', 'age', 'sex', 'help', 'hammer', 'chimpanzee']
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input df: {missing}")

    # Convert numeric columns
    df['nuts_opened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Drop rows with missing numerics or nonpositive duration
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age', 'sex', 'help'])
    df = df[df['seconds'] > 0]

    # Binary encoding for sex: male = 1, female = 0
    df['Sex_M'] = df['sex'].astype(str).str.strip().str.lower().map({
        'm': 1, 'male': 1,
        'f': 0, 'female': 0
    })

    # Binary encoding for help: yes = 1, no = 0
    df['Help'] = df['help'].astype(str).str.strip().str.lower().map({
        'y': 1, 'yes': 1, 'true': 1,
        'n': 0, 'no': 0, 'false': 0
    })

    # Drop rows where mapping produced NA (unexpected categories)
    df = df.dropna(subset=['Sex_M', 'Help'])

    # Compute raw efficiency (nuts per second) and stabilized log-efficiency
    df['Efficiency'] = df['nuts_opened'] / df['seconds']
    eps = 1e-6
    df['LogEfficiency'] = np.log(df['Efficiency'] + eps)

    # Ensure categorical types for hammer and chimpanzee (grouping variable)
    df['hammer'] = df['hammer'].astype('category')
    df['chimpanzee'] = df['chimpanzee'].astype('category')

    # Reset index (clean output)
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a linear mixed-effects model predicting LogEfficiency from age, sex (Sex_M), and Help,
    controlling for hammer type and including a random intercept for chimpanzee.

    Returns the fitted model object (statsmodels MixedLMResults).
    """
    import statsmodels.formula.api as smf

    # Work on a copy to avoid side effects
    df = df.copy()

    # Check the required columns are present
    req = ['LogEfficiency', 'age', 'Sex_M', 'Help', 'hammer', 'chimpanzee']
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Formula: fixed effects for age, Sex_M, Help and categorical hammer; random intercept for chimpanzee
    formula = 'LogEfficiency ~ age + Sex_M + Help + C(hammer)'

    # Fit mixed linear model with random intercept for chimpanzee
    md = smf.mixedlm(formula, df, groups=df['chimpanzee'])
    # Use maximum likelihood (reml=False) for comparability and inference
    mdf = md.fit(reml=False)

    # Return the fitted model object; the caller can inspect mdf.summary() or mdf.params
    return mdf


