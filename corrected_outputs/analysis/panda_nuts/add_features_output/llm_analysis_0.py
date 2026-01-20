from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/panda_nuts/add_features_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset to produce the variables needed for modeling.

    Produces:
      - Efficiency_npm: nuts opened per minute (nuts_opened / seconds * 60)
      - log_Efficiency: log1p(Efficiency_npm) to handle zeros/skew
      - sex_male: binary indicator male=1, female=0
      - help_yes: binary indicator help received yes=1, no=0
      - age_help: interaction term (age * help_yes)
      - hammer_*: dummy variables for hammer types (drop_first=True)

    Drops rows with missing or invalid data required for the analysis.
    """
    df = df.copy()

    # Drop rows missing core variables
    required = ['nuts_opened', 'seconds', 'age', 'sex', 'help', 'chimpanzee']
    df = df.dropna(subset=required)

    # Remove non-positive session durations
    df = df[df['seconds'] > 0]

    # Efficiency: nuts per minute
    df['Efficiency_npm'] = df['nuts_opened'].astype(float) / df['seconds'].astype(float) * 60.0

    # Log-transformed efficiency to stabilize variance and handle zeros (log1p)
    df['log_Efficiency'] = np.log1p(df['Efficiency_npm'])

    # Sex -> binary (male = 1, female = 0). Normalize strings and map.
    df['sex_male'] = (
        df['sex'].astype(str)
          .str.strip()
          .str.lower()
          .map({'m': 1, 'f': 0})
    )

    # Help -> binary (yes = 1, no = 0). Normalize strings and map.
    df['help_yes'] = (
        df['help'].astype(str)
          .str.strip()
          .str.lower()
          .map({'y': 1, 'n': 0})
    )

    # Drop rows where mapping failed for sex/help
    df = df.dropna(subset=['sex_male', 'help_yes'])

    # Interaction: age * help
    df['age_help'] = df['age'].astype(float) * df['help_yes'].astype(float)

    # Hammer dummies (categorical control). Use drop_first to avoid multicollinearity.
    # Normalize hammer values to strings first.
    hammer_dummies = pd.get_dummies(df['hammer'].astype(str).str.strip(), prefix='hammer', drop_first=True)
    # Concatenate dummies into dataframe
    if not hammer_dummies.empty:
        df = pd.concat([df, hammer_dummies], axis=1)

    # Final dropna for model columns (ensure no remaining missing in key columns)
    model_cols = ['Efficiency_npm', 'log_Efficiency', 'age', 'sex_male', 'help_yes', 'age_help', 'chimpanzee']
    # include hammer dummies if present
    model_cols += [c for c in df.columns if c.startswith('hammer_')]
    df = df.dropna(subset=model_cols)

    # Ensure chimpanzee grouping column is of an appropriate dtype
    df['chimpanzee'] = df['chimpanzee'].astype(str)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a linear mixed-effects model predicting log-transformed nut-cracking efficiency.

    Model specification (mixed effects):
      Endog: log_Efficiency
      Exog: constant + age + sex_male + help_yes + age_help + hammer dummies
      Random effects: random intercept by chimpanzee

    Returns the fitted MixedLMResults object.
    """
    df = df.copy()

    # Select hammer dummies if present
    hammer_cols = [c for c in df.columns if c.startswith('hammer_')]

    exog_cols = ['age', 'sex_male', 'help_yes', 'age_help'] + hammer_cols

    # Ensure exog exists and has no missing values
    exog = df[exog_cols].astype(float)
    exog = sm.add_constant(exog, has_constant='add')

    endog = df['log_Efficiency'].astype(float)

    # Grouping variable for random intercepts
    groups = df['chimpanzee']

    # Fit linear mixed-effects model with a random intercept for chimpanzee
    # Use reml=False for likelihood-based comparison if needed
    model = sm.MixedLM(endog, exog, groups=groups)
    results = model.fit(reml=False)

    # Print brief summary and return results object
    print(results.summary())
    return results


