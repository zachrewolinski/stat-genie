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
    Transform raw dataset into a dataframe with the columns required for modeling.

    Expected original columns (per schema):
      - feature1: ID of individual
      - feature2: Age in years
      - feature3: Sex ('f' or 'm')
      - feature4: Hammer type (categorical)
      - feature5: Number of nuts opened in session
      - feature6: Duration of session in seconds
      - feature7: Received help from another chimpanzee ('y' or 'N')

    Produces columns used in the model:
      - ID (string)
      - Age (float)
      - Sex_F (0/1)
      - Help (0/1)
      - HammerType (categorical)
      - NutsOpened (float)
      - SessionDuration (float)
      - Efficiency (nuts per second)
      - Eff_log (log(Efficiency + eps))
    """
    df = df.copy()

    # Standardize column names from schema
    # Keep original names available in case columns already present
    if 'feature1' in df.columns:
        df = df.rename(columns={'feature1': 'ID'})
    if 'feature2' in df.columns:
        df = df.rename(columns={'feature2': 'Age'})
    if 'feature3' in df.columns:
        df = df.rename(columns={'feature3': 'Sex'})
    if 'feature4' in df.columns:
        df = df.rename(columns={'feature4': 'HammerType'})
    if 'feature5' in df.columns:
        df = df.rename(columns={'feature5': 'NutsOpened'})
    if 'feature6' in df.columns:
        df = df.rename(columns={'feature6': 'SessionDuration'})
    if 'feature7' in df.columns:
        df = df.rename(columns={'feature7': 'HelpRaw'})

    # Keep only rows with the necessary columns non-missing
    needed = ['ID', 'Age', 'Sex', 'HammerType', 'NutsOpened', 'SessionDuration', 'HelpRaw']
    present = [c for c in needed if c in df.columns]
    if len(present) < len(needed):
        missing = list(set(needed) - set(present))
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    df = df.dropna(subset=['Age', 'Sex', 'NutsOpened', 'SessionDuration', 'HelpRaw', 'HammerType'])

    # Convert types
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['NutsOpened'] = pd.to_numeric(df['NutsOpened'], errors='coerce')
    df['SessionDuration'] = pd.to_numeric(df['SessionDuration'], errors='coerce')

    # Drop rows with non-positive or missing session durations
    df = df[df['SessionDuration'] > 0]

    # ID as string (for grouping)
    df['ID'] = df['ID'].astype(str)

    # Sex coding: female = 1, male = 0. If other categories exist, map conservatively.
    df['Sex'] = df['Sex'].astype(str).str.lower().str.strip()
    df['Sex_F'] = df['Sex'].map({'f': 1, 'female': 1, 'm': 0, 'male': 0})
    # If mapping produced NaN for unexpected labels, attempt to infer from first character
    df.loc[df['Sex_F'].isna(), 'Sex_F'] = df.loc[df['Sex_F'].isna(), 'Sex'].str.startswith('f').astype(int)

    # Help coding: 'y' or 'Y' -> 1, 'n' / 'N' -> 0
    df['HelpRaw'] = df['HelpRaw'].astype(str).str.strip()
    df['Help'] = df['HelpRaw'].str.lower().map({'y': 1, 'yes': 1, 'n': 0, 'no': 0})
    # For any unexpected labels, treat non-empty 'y'-like entries as 1, else 0
    df.loc[df['Help'].isna(), 'Help'] = df.loc[df['Help'].isna(), 'HelpRaw'].apply(lambda x: 1 if str(x).lower().startswith('y') else 0)

    # Hammer type as categorical
    df['HammerType'] = df['HammerType'].astype(str).str.strip()

    # Efficiency: nuts opened per second
    df['Efficiency'] = df['NutsOpened'] / df['SessionDuration']

    # Log-transform with small epsilon to avoid log(0)
    eps = 1e-6
    df['Eff_log'] = np.log(df['Efficiency'] + eps)

    # Keep only columns needed for modeling plus a few diagnostics
    keep_cols = ['ID', 'Age', 'Sex_F', 'Help', 'HammerType', 'NutsOpened', 'SessionDuration', 'Efficiency', 'Eff_log']
    df = df[keep_cols]

    # Final drop of any rows with missing values in modeling columns
    df = df.dropna(subset=['Age', 'Sex_F', 'Help', 'HammerType', 'Eff_log'])

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a linear mixed-effects model predicting logged efficiency (Eff_log) from Age, Sex, and Help,
    controlling for HammerType and including a random intercept for ID.

    Model (formula): Eff_log ~ Age + Sex_F + Help + C(HammerType)
    Random effects: random intercept for ID

    Returns the fitted MixedLMResults object.
    """
    # Ensure the required columns are present
    required = ['Eff_log', 'Age', 'Sex_F', 'Help', 'HammerType', 'ID']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Transformed dataframe is missing required columns for modeling: {missing}")

    # Use statsmodels mixed linear model with formula interface
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Build formula: include HammerType as categorical
    formula = 'Eff_log ~ Age + Sex_F + Help + C(HammerType)'

    # Fit the mixed-effects model with random intercept by ID
    md = smf.mixedlm(formula, data=df, groups=df['ID'])
    mdf = md.fit(reml=False, method='lbfgs')

    # Print and return results
    try:
        print(mdf.summary())
    except Exception:
        # some environments may not support pretty summaries; return the params as fallback
        print('Fixed effects coefficients:')
        print(mdf.params)

    return mdf


