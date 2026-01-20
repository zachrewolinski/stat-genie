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
    # Work on a copy
    df = df.copy()

    # Drop rows missing critical columns
    df = df.dropna(subset=['feature1','feature2','feature3','feature5','feature6','feature7'])

    # Rename / cast columns to clear names used downstream
    df['SubjectID'] = pd.to_numeric(df['feature1'], errors='coerce').astype('Int64')
    df['Age'] = pd.to_numeric(df['feature2'], errors='coerce')
    df['Sex'] = df['feature3'].astype(str).str.lower().str.strip()
    df['HammerType'] = df['feature4'].astype(str)
    df['NutsOpened'] = pd.to_numeric(df['feature5'], errors='coerce')
    df['Duration_s'] = pd.to_numeric(df['feature6'], errors='coerce')
    df['Help'] = df['feature7'].astype(str).str.lower().str.strip()

    # Drop rows that became NA after coercion
    df = df.dropna(subset=['SubjectID','Age','NutsOpened','Duration_s'])

    # Compute efficiency: nuts per minute
    # Avoid division by zero
    df = df[df['Duration_s'] > 0]
    df['Efficiency'] = df['NutsOpened'] / (df['Duration_s'] / 60.0)

    # Binary coding for sex: Male = 1, Female = 0 (fallback: unknown -> 0)
    df['Sex_M'] = df['Sex'].map({'m': 1, 'male': 1, 'f': 0, 'female': 0}).fillna(0).astype(int)

    # Binary coding for Help: yes -> 1, no -> 0 (handle common variants)
    df['Help_Y'] = df['Help'].map({'y': 1, 'yes': 1, 'n': 0, 'no': 0}).fillna(0).astype(int)

    # Center age for interpretability and interactions
    df['Age_c'] = df['Age'] - df['Age'].mean()

    # Keep only the columns required for modeling and inspection
    final_cols = ['SubjectID', 'Age', 'Age_c', 'Sex', 'Sex_M', 'HammerType', 'NutsOpened', 'Duration_s', 'Help', 'Help_Y', 'Efficiency']
    df = df[final_cols]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    import statsmodels.formula.api as smf

    # Work on a copy to avoid modifying the original
    df = df.copy()

    # Ensure categorical control is treated as category
    df['HammerType'] = df['HammerType'].astype('category')

    # Fit a linear mixed-effects model with random intercepts for SubjectID
    # Formula includes main effects for centered age, sex, help, an interaction Age_c:Help_Y,
    # and controls for HammerType (categorical). The random intercept accounts for repeated measures.
    formula = 'Efficiency ~ Age_c + Sex_M + Help_Y + Age_c:Help_Y + C(HammerType)'

    md = smf.mixedlm(formula, data=df, groups=df['SubjectID'])
    mdf = md.fit(reml=False)

    # Return the fitted model object (has summary(), params, conf_int(), etc.)
    return mdf


