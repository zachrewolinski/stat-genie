from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/panda_nuts/anonymize_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the original dataframe to produce the variables required for modeling.

    Expected original column names (per schema):
      - feature1: ID (individual identifier)
      - feature2: Age (years)
      - feature3: Sex (categorical: 'f' or 'm')
      - feature4: Hammer type (categorical)
      - feature5: Number of nuts opened in session
      - feature6: Duration of session in seconds
      - feature7: Received help (categorical: 'y' or 'N')

    Produces the following new/renamed columns used in the model:
      - ID, Age, Sex, HammerType, Nuts, DurationSec, Help
      - Sex_M (binary: 1 if male, 0 if female)
      - Help_Binary (binary: 1 if received help, 0 otherwise)
      - NutsPerMin (nuts opened per minute)
      - log_NutsPerMin (log-transformed NutsPerMin, small epsilon added)
    """
    df = df.copy()

    # Rename to meaningful column names
    df = df.rename(columns={
        'feature1': 'ID',
        'feature2': 'Age',
        'feature3': 'Sex',
        'feature4': 'HammerType',
        'feature5': 'Nuts',
        'feature6': 'DurationSec',
        'feature7': 'Help'
    })

    # Drop rows missing core variables
    df = df.dropna(subset=['Age', 'Sex', 'Nuts', 'DurationSec', 'Help'])

    # Ensure numeric types for numeric columns
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['Nuts'] = pd.to_numeric(df['Nuts'], errors='coerce')
    df['DurationSec'] = pd.to_numeric(df['DurationSec'], errors='coerce')

    # Remove rows with non-positive session durations (invalid) and any remaining NaNs
    df = df[df['DurationSec'] > 0]
    df = df.dropna(subset=['Age', 'Nuts', 'DurationSec'])

    # Normalize categorical strings
    df['Sex'] = df['Sex'].astype(str).str.lower().str.strip()
    df['HammerType'] = df['HammerType'].astype(str).str.strip()
    df['Help'] = df['Help'].astype(str).str.lower().str.strip()

    # Binary encoding for sex: Sex_M = 1 if male, 0 if female (treat any non-'m' as female for safety)
    df['Sex_M'] = (df['Sex'] == 'm').astype(int)

    # Binary encoding for help: 1 if starts with 'y' (yes), else 0
    df['Help_Binary'] = df['Help'].apply(lambda x: 1 if isinstance(x, str) and x.startswith('y') else 0).astype(int)

    # Efficiency metric: nuts per minute
    df['NutsPerMin'] = df['Nuts'] / df['DurationSec'] * 60.0

    # Log-transform the efficiency. Add a tiny epsilon to avoid log(0).
    eps = 1e-6
    df['log_NutsPerMin'] = np.log(df['NutsPerMin'] + eps)

    # Keep only columns needed for analysis (but do not drop others unnecessarily)
    # Ensure ID is present and of a suitable type
    df['ID'] = df['ID'].astype(str)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> object:
    """
    Fit a mixed-effects model predicting log-transformed nut-cracking efficiency.

    Model specification (fixed effects):
      - Age
      - Sex_M (male indicator)
      - Help_Binary (received help indicator)
      - Interactions: Sex_M * Help_Binary and Age * Help_Binary
      - C(HammerType) included as categorical fixed effect to control for hammer differences

    Random effects:
      - Random intercept for ID (individual) to account for repeated measures

    Returns the fitted mixed-effects model result object (statsmodels).
    Prints a summary before returning.
    """
    import statsmodels.formula.api as smf

    # Copy to avoid side-effects
    df = df.copy()

    # Ensure required columns exist
    required = ['log_NutsPerMin', 'Age', 'Sex_M', 'Help_Binary', 'HammerType', 'ID']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Formula: include main effects and interactions
    # C(HammerType) treats hammer as categorical control
    formula = 'log_NutsPerMin ~ Age + Sex_M * Help_Binary + Age * Help_Binary + C(HammerType)'

    # Fit mixed-effects model with random intercept by ID
    md = smf.mixedlm(formula, df, groups=df['ID'])
    mdf = md.fit(reml=False)

    # Print and return the fitted model object
    print(mdf.summary())
    return mdf


