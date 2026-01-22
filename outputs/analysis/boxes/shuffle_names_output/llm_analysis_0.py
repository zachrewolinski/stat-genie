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
    Transform the raw dataset into a dataframe ready for modeling. Assumptions based on dataset schema:
      - 'majority_first' encodes the child's choice: 1 = undemonstrated/unchosen option, 2 = majority option, 3 = minority option.
      - 'culture' column contains the child's age in years (values 4-14 in schema), while the column named 'age' encodes whether the majority was demonstrated first (0/1).
      - 'y' is the site ID.

    The function will:
      - Create binary outcome ChooseMajority (1 if choice == 2 else 0).
      - Create ChooseSocial (for exploratory use): 1 if chose either majority or minority (2 or 3).
      - Create Age, AgeCentered, AgeGroup, Culture (categorical), Gender, MajorityFirst, SiteID.
      - Drop rows missing any of the columns required for the primary model.
    """
    # Copy to avoid modifying original
    df = df.copy()

    # Standardize expected column names and types
    # According to provided schema notes: 'majority_first' = choice; 'culture' actually contains ages; 'age' encodes majority-first; 'y' is site id
    # Ensure columns exist
    expected = ['majority_first', 'gender', 'culture', 'age', 'y']
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe is missing expected columns: {missing}")

    # Create Choice and binary outcomes
    df['Choice'] = pd.to_numeric(df['majority_first'], errors='coerce').astype('Int64')
    # Binary: chose the majority option (Choice == 2)
    df['ChooseMajority'] = (df['Choice'] == 2).astype('Int64')
    # Binary: chose any demonstrated option (majority or minority)
    df['ChooseSocial'] = df['Choice'].isin([2, 3]).astype('Int64')

    # Age: use column 'culture' (schema indicates values 4-14)
    df['Age'] = pd.to_numeric(df['culture'], errors='coerce')

    # MajorityFirst (order) is in the column named 'age' according to schema inconsistencies (0/1)
    df['MajorityFirst'] = pd.to_numeric(df['age'], errors='coerce').astype('Int64')

    # Gender: keep numeric coding 1=girl, 2=boy
    df['Gender'] = pd.to_numeric(df['gender'], errors='coerce').astype('Int64')

    # SiteID and Culture: use 'y' as site identifier; create Culture as categorical site-level variable
    df['SiteID'] = pd.to_numeric(df['y'], errors='coerce').astype('Int64')
    df['Culture'] = df['SiteID'].astype('category')

    # Center age for better interpretability and interaction stability
    df['AgeCentered'] = df['Age'] - df['Age'].mean()

    # Create coarse developmental AgeGroup for descriptive analyses: Early (4-6), Middle (7-9), Late (10+)
    bins = [3.5, 6.5, 9.5, 14.5]
    labels = ['Early', 'Middle', 'Late']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, include_lowest=True)

    # Drop rows with missing values in the primary variables used in modeling
    required_for_model = ['ChooseMajority', 'Age', 'Gender', 'MajorityFirst', 'SiteID']
    df = df.dropna(subset=required_for_model)

    # Cast final columns to standard dtypes
    df['ChooseMajority'] = df['ChooseMajority'].astype(int)
    df['Gender'] = df['Gender'].astype(int)
    df['MajorityFirst'] = df['MajorityFirst'].astype(int)
    df['SiteID'] = df['SiteID'].astype(int)

    # Keep only the columns needed for modeling and sensible diagnostics (but return full df; model function will use needed columns)
    keep_cols = [
        'Choice', 'ChooseMajority', 'ChooseSocial', 'Age', 'AgeCentered', 'AgeGroup',
        'Gender', 'MajorityFirst', 'SiteID', 'Culture'
    ]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = pd.NA

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a primary statistical model testing whether children's choice of the majority option
    varies with age and across cultural sites, controlling for gender and order (MajorityFirst).

    Primary model: logistic regression (binomial GLM) predicting ChooseMajority.
    Key predictor terms: AgeCentered, Culture (categorical), and their interaction AgeCentered:C(Culture).
    Controls: Gender, MajorityFirst.

    We cluster standard errors by SiteID to account for within-site dependence.

    Returns the fitted model object with cluster-robust covariances.
    """
    import statsmodels.formula.api as smf

    # Ensure needed columns exist
    required = ['ChooseMajority', 'AgeCentered', 'Culture', 'Gender', 'MajorityFirst', 'SiteID']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Transformed dataframe is missing required columns for modeling: {missing}")

    # Formula: main effects + interaction between age and culture
    # C(Culture) treats Culture as categorical
    formula = 'ChooseMajority ~ AgeCentered * C(Culture) + Gender + MajorityFirst'

    # Fit binomial GLM
    glm_mod = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
    glm_res = glm_mod.fit()

    # Obtain cluster-robust standard errors grouped by SiteID
    try:
        glm_res_clust = glm_res.get_robustcov_results(cov_type='cluster', groups=df['SiteID'])
    except Exception:
        # Fallback: return original glm_res if clustering fails
        print('Warning: clustering by SiteID failed; returning unclustered estimates')
        glm_res_clust = glm_res

    # Print summary for quick inspection
    print(glm_res_clust.summary())

    return glm_res_clust


