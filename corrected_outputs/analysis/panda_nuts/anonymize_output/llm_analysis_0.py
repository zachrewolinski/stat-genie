from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/panda_nuts/anonymize_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the raw dataset into a dataframe ready for modeling.

    Produces the following columns required by the model:
      - id: individual ID (feature1)
      - age: age in years (feature2)
      - age_c: centered age
      - sex_male: 1 if sex == 'm', 0 if sex == 'f'
      - help_yes: 1 if received help (feature7 indicates yes), 0 otherwise
      - efficiency: nuts opened per minute (winsorized at 99th percentile)
      - hammer_Q, hammer_G: dummy variables for hammer type (reference = 'wood')

    Rows with non-positive session duration or missing key fields are dropped.
    """
    df = df.copy()

    # Rename columns to meaningful names
    df = df.rename(columns={
        'feature1': 'id',
        'feature2': 'age',
        'feature3': 'sex',
        'feature4': 'hammer',
        'feature5': 'nuts_opened',
        'feature6': 'duration_s',
        'feature7': 'help'
    })

    # Drop rows with missing essential values
    df = df.dropna(subset=['id', 'age', 'sex', 'hammer', 'nuts_opened', 'duration_s', 'help'])

    # Ensure correct dtypes
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['nuts_opened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    df['duration_s'] = pd.to_numeric(df['duration_s'], errors='coerce')

    # Remove rows with non-positive duration (can't compute rate)
    df = df[df['duration_s'] > 0].copy()

    # Compute efficiency: nuts opened per minute
    df['efficiency'] = df['nuts_opened'] / (df['duration_s'] / 60.0)

    # Replace infinite or NaN efficiencies (just in case) and drop them
    df.loc[~np.isfinite(df['efficiency']), 'efficiency'] = np.nan
    df = df.dropna(subset=['efficiency'])

    # Winsorize efficiency at 99th percentile to reduce influence of extreme outliers
    upper = np.percentile(df['efficiency'], 99)
    df['efficiency'] = df['efficiency'].clip(upper=upper)

    # Create sex binary: 1 if male ('m' or 'M'), 0 if female ('f' or 'F')
    df['sex_str'] = df['sex'].astype(str).str.strip().str.lower()
    df['sex_male'] = df['sex_str'].map(lambda x: 1 if x == 'm' else (0 if x == 'f' else np.nan))

    # Standardize help coding: treat 'y','yes','Y' etc. as yes
    df['help_str'] = df['help'].astype(str).str.strip().str.lower()
    df['help_yes'] = df['help_str'].apply(lambda x: 1 if x in ['y', 'yes'] else (0 if x in ['n', 'no'] else np.nan))

    # Drop rows where sex or help could not be parsed
    df = df.dropna(subset=['sex_male', 'help_yes'])

    # Create hammer dummies. Use 'wood' as reference if present.
    df['hammer_str'] = df['hammer'].astype(str).str.strip()
    hammer_dummies = pd.get_dummies(df['hammer_str'], prefix='hammer')
    # keep only dummies for non-reference categories; choose 'hammer_wood' as reference if exists
    if 'hammer_wood' in hammer_dummies.columns:
        # drop reference column
        hammer_dummies = hammer_dummies.drop(columns=['hammer_wood'])
    # If 'wood' isn't present in data, drop one column to avoid collinearity (drop first)
    if hammer_dummies.shape[1] > 0:
        # No further action needed; use these dummies as controls
        pass

    df = pd.concat([df, hammer_dummies.reset_index(drop=True)], axis=1)

    # Center age for improved interpretability
    df['age_c'] = df['age'] - df['age'].mean()

    # Keep only the columns needed for modeling (plus some diagnostics)
    # Identify which hammer dummy columns are present
    hammer_cols = [c for c in df.columns if c.startswith('hammer_')]

    keep_cols = ['id', 'age', 'age_c', 'sex_male', 'help_yes', 'efficiency'] + hammer_cols
    df = df[keep_cols]

    # Reset index
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fits a linear mixed effects model predicting nut-cracking efficiency from age, sex, and help,
    controlling for hammer type and including a random intercept for individual chimpanzee.

    Model (fixed effects):
      efficiency ~ age_c + sex_male + help_yes + hammer_Q + hammer_G + (age_c:help_yes)

    Random effects:
      random intercept by id

    Returns the fitted model result object (MixedLMResults) and prints a brief summary.
    """
    import statsmodels.api as sm

    # Ensure hammer dummies existence in the dataframe; if missing, add columns with zeros
    hammer_possible = ['hammer_Q', 'hammer_G']
    for hc in hammer_possible:
        if hc not in df.columns:
            df[hc] = 0

    # Build formula. Include an interaction between age and help to test whether the effect of age
    # on efficiency differs depending on whether the chimp received help.
    formula = 'efficiency ~ age_c + sex_male + help_yes + age_c:help_yes + hammer_Q + hammer_G'

    # Fit a linear mixed effects model with random intercept for each individual
    # Use REML=False for easier comparisons if needed
    md = sm.MixedLM.from_formula(formula, groups='id', data=df)
    mdf = md.fit(reml=False)

    # Print summary
    print(mdf.summary())

    return mdf


