from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/anonymize_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the tidy dataframe used for modeling.

    Expects original columns:
      - feature1: outcome code (1=unchosen option, 2=majority option, 3=minority option)
      - feature2: gender code (1=girl, 2=boy)
      - feature3: age in years (4-14)
      - feature4: whether majority was demonstrated first (0/1)
      - feature5: site ID (integer)

    Returns dataframe with columns used in modeling:
      - ChoiceRaw, Age, Male, MajorityFirst, Site,
      - ChoseDemonstrated (0/1), ChoseMajorityAmongDem (0/1 or np.nan),
      - Age_centered, AgeGroup
    """
    df = df.copy()

    # Rename raw columns for clarity
    df = df.rename(columns={
        'feature1': 'ChoiceRaw',
        'feature2': 'GenderRaw',
        'feature3': 'Age',
        'feature4': 'MajorityFirst',
        'feature5': 'Site'
    })

    # Ensure correct dtypes
    df['ChoiceRaw'] = pd.to_numeric(df['ChoiceRaw'], errors='coerce')
    df['GenderRaw'] = pd.to_numeric(df['GenderRaw'], errors='coerce')
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['MajorityFirst'] = pd.to_numeric(df['MajorityFirst'], errors='coerce')
    df['Site'] = df['Site'].astype('category')

    # Drop rows with essential missing values (outcome, age, site)
    df = df.dropna(subset=['ChoiceRaw', 'Age', 'Site'])

    # Map outcome codes to labels (optional) and create analysis variables
    # feature1: 1=unchosen option, 2=majority option, 3=minority option
    df['ChoiceLabel'] = df['ChoiceRaw'].map({1: 'Unchosen', 2: 'Majority', 3: 'Minority'})

    # Dependent variable 1: Did the child choose a demonstrated option (majority or minority) vs an unchosen option?
    df['ChoseDemonstrated'] = df['ChoiceRaw'].apply(lambda x: 1 if x in [2, 3] else 0).astype('int')

    # Dependent variable 2 (conditional): Among those who selected a demonstrated option, did they choose the majority (1) or the minority (0)?
    # Set to NaN for those who did not choose a demonstrated option so downstream models can subset.
    df['ChoseMajorityAmongDem'] = df['ChoiceRaw'].apply(lambda x: 1 if x == 2 else (0 if x == 3 else np.nan))

    # Controls: Gender (1=girl, 2=boy) -> create Male binary
    df['Male'] = df['GenderRaw'].apply(lambda x: 1 if x == 2 else 0).astype('int')

    # Ensure MajorityFirst is binary 0/1
    df['MajorityFirst'] = df['MajorityFirst'].apply(lambda x: 1 if x == 1 else 0).astype('int')

    # Center age for interpretability and potential interactions
    df['Age_centered'] = df['Age'] - df['Age'].mean()

    # Create coarse AgeGroup categories for descriptive checks / plotting
    # 4-6, 7-9, 10-12, 13-14
    bins = [3.5, 6.5, 9.5, 12.5, 14.5]
    labels = ['4-6', '7-9', '10-12', '13-14']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, ordered=True)

    # Final check / reorder columns for clarity
    cols_keep = [
        'ChoiceRaw', 'ChoiceLabel', 'ChoseDemonstrated', 'ChoseMajorityAmongDem',
        'Age', 'Age_centered', 'AgeGroup', 'GenderRaw', 'Male', 'MajorityFirst', 'Site'
    ]
    # Some datasets may lack GenderRaw; ensure we only select existing columns
    cols_keep = [c for c in cols_keep if c in df.columns]

    return df[cols_keep]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fits two logistic regression models to answer the research question:
      Model A (social reliance): ChoseDemonstrated (0/1) ~ Age_centered * C(Site) + Male + MajorityFirst
      Model B (majority preference among demonstrated choices): ChoseMajorityAmongDem (0/1) ~ Age_centered * C(Site) + Male + MajorityFirst

    Returns a dictionary with fitted model objects and summary tables.
    """
    import statsmodels.formula.api as smf

    results = {}

    # Ensure categorical Site treated as category in the dataframe
    if 'Site' in df.columns:
        df['Site'] = df['Site'].astype('category')

    # Model A: reliance on social information (binary logistic regression)
    # We include an interaction between age and site to test whether developmental changes differ by culture.
    formula_a = 'ChoseDemonstrated ~ Age_centered * C(Site) + Male + MajorityFirst'
    try:
        model_a = smf.logit(formula=formula_a, data=df).fit(disp=False)
        results['model_social_reliance'] = model_a
        results['model_social_reliance_summary'] = model_a.summary()
        # Also store odds ratios and 95% CI for easier interpretation
        params = model_a.params
        conf = model_a.conf_int()
        or_table = pd.DataFrame({
            'OR': np.exp(params),
            'CI_lower': np.exp(conf[0]),
            'CI_upper': np.exp(conf[1])
        })
        results['model_social_reliance_or'] = or_table
    except Exception as e:
        results['model_social_reliance_error'] = str(e)

    # Model B: majority preference conditional on choosing a demonstrated option
    df_demonstrated = df[df['ChoseDemonstrated'] == 1].copy()

    if df_demonstrated.shape[0] < 20:
        # If too few cases, warn the user and still attempt to fit if possible
        results['warning_demonstrated_sample_size'] = (
            f"Small sample for majority-vs-minority model: n={df_demonstrated.shape[0]}"
        )

    formula_b = 'ChoseMajorityAmongDem ~ Age_centered * C(Site) + Male + MajorityFirst'
    try:
        model_b = smf.logit(formula=formula_b, data=df_demonstrated).fit(disp=False)
        results['model_majority_preference'] = model_b
        results['model_majority_preference_summary'] = model_b.summary()
        params_b = model_b.params
        conf_b = model_b.conf_int()
        or_table_b = pd.DataFrame({
            'OR': np.exp(params_b),
            'CI_lower': np.exp(conf_b[0]),
            'CI_upper': np.exp(conf_b[1])
        })
        results['model_majority_preference_or'] = or_table_b
    except Exception as e:
        results['model_majority_preference_error'] = str(e)

    # Additional outputs that are helpful for interpretation
    results['n_total'] = int(df.shape[0])
    results['n_demonstrated'] = int(df_demonstrated.shape[0])
    results['choice_counts'] = df['ChoiceLabel'].value_counts(dropna=False).to_dict()

    return results


