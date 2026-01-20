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
    Transform the raw dataset into a dataframe suitable for modeling.

    Steps:
    - Copy input dataframe to avoid inplace modification.
    - Ensure numeric columns are numeric and drop rows with missing essential values.
    - Remove sessions with non-positive duration.
    - Compute Efficiency = nuts_opened per minute (nuts_opened * 60 / seconds).
    - Normalize categorical columns (sex, help, hammer) and ensure chimpanzee ID is integer.

    Returns the transformed dataframe containing at least the columns:
    ['chimpanzee', 'age', 'sex', 'help', 'hammer', 'nuts_opened', 'seconds', 'Efficiency']
    """
    df = df.copy()

    # Ensure the essential columns exist; if not, this will raise a KeyError for the user to diagnose
    required_cols = ['chimpanzee', 'age', 'sex', 'help', 'hammer', 'nuts_opened', 'seconds']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in input dataframe: {missing}")

    # Coerce numeric columns
    df['nuts_opened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['chimpanzee'] = pd.to_numeric(df['chimpanzee'], errors='coerce')

    # Drop rows with missing essential numeric values
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age', 'chimpanzee'])

    # Remove non-positive durations (seconds must be > 0)
    df = df[df['seconds'] > 0]

    # Compute efficiency: nuts opened per minute
    df['Efficiency'] = df['nuts_opened'] * 60.0 / df['seconds']

    # Normalize categorical variables. Handle inconsistent capitalization (e.g., 'y' and 'N')
    # Sex: map common labels to 'M' and 'F'
    df['sex'] = df['sex'].astype(str).str.strip().str.lower().map({
        'm': 'M', 'male': 'M', 'f': 'F', 'female': 'F'
    }).fillna(df['sex'].astype(str))
    df['sex'] = df['sex'].astype('category')

    # Help: map to 'yes' / 'no'
    df['help'] = df['help'].astype(str).str.strip().str.lower().map({
        'y': 'yes', 'yes': 'yes', 'n': 'no', 'no': 'no'
    }).fillna(df['help'].astype(str))
    df['help'] = df['help'].astype('category')

    # Hammer: ensure categorical
    df['hammer'] = df['hammer'].astype(str).str.strip()
    df['hammer'] = df['hammer'].astype('category')

    # Ensure chimpanzee is integer id
    df['chimpanzee'] = df['chimpanzee'].astype(int)

    # Final drop in case mapping created any NA categories
    df = df.dropna(subset=['sex', 'help', 'hammer'])

    # Reorder columns so the main model columns are easy to find (optional)
    cols = ['chimpanzee', 'age', 'sex', 'help', 'hammer', 'nuts_opened', 'seconds', 'Efficiency']
    other_cols = [c for c in df.columns if c not in cols]
    df = df[cols + other_cols]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a mixed-effects linear model predicting nut-cracking efficiency.

    Model specification:
    - Dependent variable: Efficiency (nuts per minute).
    - Fixed effects: age (continuous), sex (categorical), help (categorical), hammer (categorical).
    - Interactions: age:help and sex:help to test whether receiving help changes how age or sex relate to efficiency.
    - Random effects: random intercept for chimpanzee to account for repeated measures by individual.

    Returns the fitted MixedLMResults object. Prints a brief summary.
    """
    import statsmodels.formula.api as smf

    # Ensure required columns present
    required = ['Efficiency', 'age', 'sex', 'help', 'hammer', 'chimpanzee']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for modeling: {missing}")

    # Formula with categorical indicators using C()
    formula = (
        "Efficiency ~ age + C(sex) + C(help) + age:C(help) + C(sex):C(help) + C(hammer)"
    )

    # Fit mixed effects model with random intercept per chimpanzee
    md = smf.mixedlm(formula, data=df, groups=df['chimpanzee'], re_formula='~1')
    mdf = md.fit(reml=False)

    # Print summary and return the fitted model
    print(mdf.summary())
    return mdf


