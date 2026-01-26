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
    Transform raw dataset to a dataframe suitable for modeling.

    - Renames columns to clear variable names used in the model.
    - Drops rows with missing essential fields.
    - Converts types (numeric, categorical mapping).
    - Computes NutEfficiency = NutsOpened / DurationSec.
    - Creates hammer-type dummy variables (prefix 'Hammer_', drop_first=True).

    Returns:
        Transformed pandas DataFrame with columns (at minimum):
        ['ID', 'Age', 'Sex', 'ReceivedHelp', 'HammerType', 'NutsOpened', 'DurationSec', 'NutEfficiency', 'Hammer_*']
    """
    df = df.copy()

    # Rename incoming feature columns to meaningful names
    df = df.rename(columns={
        'feature1': 'ID',
        'feature2': 'Age',
        'feature3': 'Sex',
        'feature4': 'HammerType',
        'feature5': 'NutsOpened',
        'feature6': 'DurationSec',
        'feature7': 'ReceivedHelp'
    })

    # Drop rows missing core measurement fields
    df = df.dropna(subset=['Age', 'Sex', 'HammerType', 'NutsOpened', 'DurationSec', 'ReceivedHelp'])

    # Ensure numeric columns are numeric
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['NutsOpened'] = pd.to_numeric(df['NutsOpened'], errors='coerce')
    df['DurationSec'] = pd.to_numeric(df['DurationSec'], errors='coerce')

    # Drop rows where conversion failed
    df = df.dropna(subset=['Age', 'NutsOpened', 'DurationSec'])

    # Remove impossible/invalid durations
    df = df[df['DurationSec'] > 0]

    # Compute the dependent variable: nuts opened per second
    df['NutEfficiency'] = df['NutsOpened'] / df['DurationSec']

    # Map Sex to binary numeric: male = 1, female = 0
    # Accept common variants by lowercasing first
    df['Sex'] = df['Sex'].astype(str).str.strip().str.lower().map({
        'm': 1, 'male': 1,
        'f': 0, 'female': 0
    })

    # Map ReceivedHelp to binary numeric: yes = 1, no = 0
    df['ReceivedHelp'] = df['ReceivedHelp'].astype(str).str.strip().str.lower().map({
        'y': 1, 'yes': 1, 'true': 1, '1': 1,
        'n': 0, 'no': 0, 'false': 0, '0': 0
    })

    # Drop rows where mapping produced NA (unexpected categories)
    df = df.dropna(subset=['Sex', 'ReceivedHelp'])

    # Ensure ID is treated as categorical/grouping variable
    df['ID'] = df['ID'].astype(int)

    # Create dummy variables for hammer type, drop_first to avoid collinearity.
    # Dummy columns will be named like 'Hammer_<level>' (except dropped first level).
    hammer_dummies = pd.get_dummies(df['HammerType'].astype(str), prefix='Hammer', drop_first=True)
    df = pd.concat([df.reset_index(drop=True), hammer_dummies.reset_index(drop=True)], axis=1)

    # Final dataframe ready for modeling
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a linear mixed-effects model predicting NutEfficiency from Age, Sex, and ReceivedHelp,
    controlling for hammer type (fixed effects) and with a random intercept for individual ID.

    Returns:
        Fitted MixedLMResults object (statsmodels) and prints the model summary.
    """
    import statsmodels.formula.api as smf

    # Work on a copy
    data = df.copy()

    # Identify hammer dummy columns created during transform (those starting with 'Hammer_')
    hammer_cols = [c for c in data.columns if c.startswith('Hammer_')]

    # Build formula: NutEfficiency ~ Age + Sex + ReceivedHelp [+ hammer dummies]
    base_terms = ['Age', 'Sex', 'ReceivedHelp']
    all_terms = base_terms + hammer_cols
    formula = 'NutEfficiency ~ ' + ' + '.join(all_terms)

    # Fit mixed effects model with random intercept for ID
    # Use groups=data['ID'] to model repeated measures per individual
    md = smf.mixedlm(formula, data, groups=data['ID'])
    mdf = md.fit(reml=False)

    # Print and return fitted model object
    print(mdf.summary())
    return mdf


