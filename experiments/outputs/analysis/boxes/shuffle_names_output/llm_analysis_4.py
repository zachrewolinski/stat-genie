from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/shuffle_names_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into analysis-ready dataframe with the following columns (exact names used in modeling):
      - Choice: categorical label of the child's choice ('unchosen','majority','minority')
      - SocialUse: binary (1 if child chose a demonstrated option [majority or minority], 0 if chose undemonstrated)
      - MajorityChoice: binary (1 if child chose the majority option, 0 if chose minority). For rows where Choice is 'unchosen', MajorityChoice will be 0 but these rows should be ignored in the majority-preference model (we subset later).
      - Age: numeric age in years (in the provided data this is stored in the 'culture' column based on the schema metadata)
      - Gender: binary (0 = girl, 1 = boy) mapped from provided 'gender' column
      - MajorityDemoFirst: binary order indicator (mapped from provided 'age' column in the raw schema where 0/1 indicates whether majority was demonstrated first)
      - Site: categorical site/culture id (from column 'y')

    Notes on schema inconsistencies: The provided dataset schema appears to have mismatched descriptions for some columns. Based on value ranges and descriptions we assume:
      - 'culture' column contains the child's age in years (values 4-14)
      - 'age' column contains the binary indicator of whether the majority demonstration was shown first (0/1)
      - 'y' is the site id / culture id

    The function will drop rows with missing values in any of the required columns.
    """
    df = df.copy()

    # Map / rename columns according to inferred semantics
    # 'culture' column appears to contain numeric ages (4-14) per schema metadata
    if 'culture' not in df.columns:
        raise KeyError("Expected column 'culture' in the input dataframe")
    df['Age'] = pd.to_numeric(df['culture'], errors='coerce')

    # 'y' is treated as site / culture id
    if 'y' not in df.columns:
        raise KeyError("Expected column 'y' in the input dataframe")
    df['Site'] = df['y'].astype('category')

    # 'age' per schema appears to be the binary indicator whether majority was demonstrated first
    if 'age' not in df.columns:
        raise KeyError("Expected column 'age' in the input dataframe")
    # Ensure binary (0/1)
    df['MajorityDemoFirst'] = pd.to_numeric(df['age'], errors='coerce').fillna(0).astype(int)

    # Map gender: schema says 1=girl, 2=boy
    if 'gender' not in df.columns:
        raise KeyError("Expected column 'gender' in the input dataframe")
    df['Gender'] = pd.to_numeric(df['gender'], errors='coerce')
    # Map to 0/1 (0 = girl, 1 = boy)
    df['Gender'] = df['Gender'].map({1: 0, 2: 1}).astype('Int64')

    # Map choice outcome from 'majority_first' column: 1 = unchosen option, 2 = majority option, 3 = minority option
    if 'majority_first' not in df.columns:
        raise KeyError("Expected column 'majority_first' in the input dataframe")
    df['majority_first'] = pd.to_numeric(df['majority_first'], errors='coerce')
    choice_map = {1: 'unchosen', 2: 'majority', 3: 'minority'}
    df['Choice'] = df['majority_first'].map(choice_map)

    # Create SocialUse: 1 if choice is majority or minority, 0 if unchosen
    df['SocialUse'] = df['Choice'].isin(['majority', 'minority']).astype(int)

    # Create MajorityChoice: among all rows mark 1 if majority chosen, 0 otherwise
    df['MajorityChoice'] = (df['Choice'] == 'majority').astype(int)

    # Keep only necessary columns for modeling
    final_cols = ['Choice', 'SocialUse', 'MajorityChoice', 'Age', 'Gender', 'MajorityDemoFirst', 'Site']
    df = df[final_cols]

    # Drop rows with missing values in any of the required model columns
    df = df.dropna(subset=['SocialUse', 'MajorityChoice', 'Age', 'Gender', 'MajorityDemoFirst', 'Site'])

    # Ensure correct dtypes
    df['Age'] = df['Age'].astype(float)
    df['Gender'] = df['Gender'].astype(int)
    df['MajorityDemoFirst'] = df['MajorityDemoFirst'].astype(int)
    df['SocialUse'] = df['SocialUse'].astype(int)
    df['MajorityChoice'] = df['MajorityChoice'].astype(int)

    # Return the cleaned/transformed dataframe used in modeling
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit the planned statistical models addressing the research question.

    Models fitted:
      1) Primary analysis (binary outcome): SocialUse (did the child rely on social information?)
         - Model: logistic regression (GLM with binomial family)
         - Predictors: Age, Site (categorical), Age x Site interaction, Gender, MajorityDemoFirst
         - Formula (as implemented): 'SocialUse ~ Age * C(Site) + Gender + MajorityDemoFirst'

      2) Secondary analysis (among children who chose a demonstrated option): MajorityChoice (did the child pick the majority rather than the minority?)
         - Model: logistic regression (GLM with binomial family) on subset df[SocialUse==1]
         - Predictors: same as above

    Returns a dictionary with the fitted model results and some descriptive summaries.
    """
    import statsmodels.api as _sm
    import statsmodels.formula.api as smf

    results = {}

    # Ensure required columns exist
    required = ['SocialUse', 'Age', 'Site', 'Gender', 'MajorityDemoFirst', 'MajorityChoice']
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Required column '{c}' not found in dataframe passed to model()")

    # Model 1: SocialUse ~ Age * Site + Gender + MajorityDemoFirst
    formula1 = 'SocialUse ~ Age * C(Site) + Gender + MajorityDemoFirst'
    model1 = smf.glm(formula=formula1, data=df, family=_sm.families.Binomial()).fit()
    results['social_use_model'] = model1

    # Model 2: Majority preference among those who used social information
    df_demo = df[df['SocialUse'] == 1].copy()
    if df_demo.shape[0] < 20:
        # too few observations for reliable site-by-age interactions; still fit, but warn
        pass
    formula2 = 'MajorityChoice ~ Age * C(Site) + Gender + MajorityDemoFirst'
    model2 = smf.glm(formula=formula2, data=df_demo, family=_sm.families.Binomial()).fit()
    results['majority_choice_model'] = model2

    # Add descriptive summaries
    desc = {
        'n_total': int(df.shape[0]),
        'n_used_social_info': int(df['SocialUse'].sum()),
        'n_sites': int(df['Site'].nunique()),
        'age_mean': float(df['Age'].mean()),
        'age_std': float(df['Age'].std())
    }
    results['descriptives'] = desc

    # Optionally return model summaries as text for quick inspection
    results['social_use_summary'] = model1.summary().as_text()
    results['majority_choice_summary'] = model2.summary().as_text()

    return results


