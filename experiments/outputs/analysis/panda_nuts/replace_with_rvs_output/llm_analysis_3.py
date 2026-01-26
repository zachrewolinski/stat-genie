from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/panda_nuts/replace_with_rvs_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy to avoid modifying original
    df = df.copy()

    # Ensure required columns are present
    required_cols = ['chimpanzee', 'age', 'sex', 'hammer', 'nuts_opened', 'seconds', 'help']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns in input dataframe: {missing}")

    # Coerce numeric columns
    df['nuts_opened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Normalize categorical text fields and coerce to string
    df['sex'] = df['sex'].astype(str).str.strip().str.lower().replace({'female':'f','male':'m'})
    df['help'] = df['help'].astype(str).str.strip().str.lower()
    # Normalize help values: map common variants to 'yes'/'no'
    df['help'] = df['help'].replace({'y': 'yes', 'yes': 'yes', 'n': 'no', 'no': 'no'})
    # Ensure hammer is a string category
    df['hammer'] = df['hammer'].astype(str).str.strip()

    # Drop rows with missing critical values
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age', 'sex', 'help'])

    # Remove sessions with non-positive seconds (invalid duration)
    df = df[df['seconds'] > 0]

    # Compute efficiency: nuts opened per second, then convert to per minute for interpretability
    df['Efficiency_per_min'] = df['nuts_opened'] / df['seconds'] * 60.0

    # Small constant to avoid log(0) issues
    eps = 1e-6
    df['log_Efficiency_per_min'] = np.log(df['Efficiency_per_min'] + eps)

    # Standardize categorical levels to expected coding for modeling
    # sex: map to 'f' and 'm' (if other values exist keep as-is)
    df['sex'] = df['sex'].replace({'female':'f','male':'m'})

    # help: ensure only 'yes'/'no' remain; drop other / ambiguous entries
    df = df[df['help'].isin(['yes', 'no'])]

    # hammer: treat as categorical, keep original labels
    df['hammer'] = df['hammer'].astype('category')

    # chimpanzee: ensure grouping variable is present and categorical
    # Keep original ids but convert to string category to avoid numeric modeling confusion
    df['chimpanzee'] = df['chimpanzee'].astype(str)

    # Reset index
    df = df.reset_index(drop=True)

    # Keep only columns needed for modeling plus original useful columns
    keep_cols = ['chimpanzee', 'age', 'sex', 'help', 'hammer', 'nuts_opened', 'seconds', 'Efficiency_per_min', 'log_Efficiency_per_min']
    df = df[keep_cols]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fits a linear mixed-effects model predicting log-transformed nut-cracking efficiency
    using fixed effects for age, sex, help, and hammer type and a random intercept for chimpanzee.

    Returns the fitted model object (statsmodels MixedLMResults) and prints a summary.
    """
    import statsmodels.formula.api as smf

    # Formula: main effects of age (continuous), sex (categorical), help (categorical), hammer (categorical)
    formula = 'log_Efficiency_per_min ~ age + C(sex) + C(help) + C(hammer)'

    # Fit mixed-effects model with random intercept for chimpanzee
    md = smf.mixedlm(formula, df, groups=df['chimpanzee'])
    # Use ML (reml=False) for easier comparison of nested models if needed
    mdf = md.fit(reml=False, method='nm')

    # Print summary for quick inspection
    print(mdf.summary())

    # Recommended further checks (not executed here): inspect residuals, influence, plot observed vs predicted

    # Return the fitted model results object for downstream use
    results = mdf
    return results


