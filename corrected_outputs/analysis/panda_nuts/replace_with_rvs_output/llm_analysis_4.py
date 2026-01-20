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
    Clean and transform the raw dataset for modeling nut-cracking efficiency.

    Produces these new columns used in the model:
      - NutsPerSec: nuts_opened divided by session duration (seconds)
      - log_NutsPerSec: log1p-transformed rate to stabilize variance and handle zeros
      - age_c: mean-centered age
      - sex: cleaned categorical sex variable (lowercase 'f'/'m')
      - help: cleaned categorical help variable ('yes'/'no')
      - hammer: ensured string/categorical hammer variable
    Also drops rows with missing or invalid critical values (seconds <= 0, missing nuts_opened, age, sex, help, chimpanzee).
    """
    df = df.copy()

    # Ensure required columns exist
    required_cols = ['chimpanzee', 'age', 'sex', 'hammer', 'nuts_opened', 'seconds', 'help']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Drop rows with missing critical values
    df = df.dropna(subset=['chimpanzee', 'age', 'sex', 'nuts_opened', 'seconds', 'help'])

    # Remove sessions with non-positive duration
    df = df[df['seconds'] > 0]

    # Ensure numeric types where expected
    df['nuts_opened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Drop any rows that became NA after coercion
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age'])

    # Compute efficiency: nuts opened per second
    df['NutsPerSec'] = df['nuts_opened'] / df['seconds']

    # Stabilize distribution: use log1p on rate (handles zeros naturally)
    # log1p(NutsPerSec) = log(1 + NutsPerSecond)
    df['log_NutsPerSec'] = np.log1p(df['NutsPerSec'])

    # Center age for interpretability
    df['age_c'] = df['age'] - df['age'].mean()

    # Clean sex column: lowercase and standardize to 'f' or 'm'
    df['sex'] = df['sex'].astype(str).str.strip().str.lower().replace({'female': 'f', 'male': 'm'})
    # Keep only expected levels; drop others
    df = df[df['sex'].isin(['f', 'm'])]

    # Clean help column: map a variety of encodings to 'yes'/'no'
    df['help'] = df['help'].astype(str).str.strip().str.lower()
    df['help'] = df['help'].map(lambda x: 'yes' if x in ['y', 'yes', 'true', '1'] else 'no')

    # Ensure hammer is string/categorical
    df['hammer'] = df['hammer'].astype(str).str.strip()

    # Ensure chimpanzee ID is categorical/int
    # keep original chimpanzee column name for grouping in model
    df['chimpanzee'] = df['chimpanzee'].astype(int)

    # Final drop in case cleaning produced NAs
    df = df.dropna(subset=['log_NutsPerSec', 'age_c', 'sex', 'help', 'hammer', 'chimpanzee'])

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a mixed-effects linear model predicting log_NutsPerSec from age, sex, and help,
    controlling for hammer type and accounting for repeated measures by chimpanzee (random intercept).

    Model formula:
      log_NutsPerSec ~ age_c + C(sex) + C(help) + age_c:C(help) + C(hammer)

    We include an age-by-help interaction to test whether the effect of receiving help differs across ages
    (e.g., younger individuals may benefit more from help).

    Returns the fitted statsmodels MixedLMResults object.
    """
    import statsmodels.formula.api as smf

    # Ensure the transformed columns exist
    required = ['log_NutsPerSec', 'age_c', 'sex', 'help', 'hammer', 'chimpanzee']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Specify formula: categorical variables wrapped in C() to force treatment coding
    formula = 'log_NutsPerSec ~ age_c + C(sex) + C(help) + age_c:C(help) + C(hammer)'

    # Fit mixed-effects model with random intercept for chimpanzee
    md = smf.mixedlm(formula, df, groups=df['chimpanzee'])
    mdf = md.fit(reml=False)

    # Return the fitted model object (contains summary, params, pvalues, etc.)
    return mdf


