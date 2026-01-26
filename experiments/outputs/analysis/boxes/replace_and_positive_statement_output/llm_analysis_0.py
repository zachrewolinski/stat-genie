from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/replace_and_positive_statement_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset into analysis-ready dataframe.

    Produces the following new columns used by the models:
      - ChoseDemonstrated: binary (1 if y in {2,3} i.e., chose majority or minority; 0 if y==1 i.e., undemonstrated)
      - ChoseMajority: binary among demonstrated choices (1 if y==2 majority, 0 if y==3 minority, NaN if y==1)
      - age_centered: age minus mean(age)
      - AgeGroup: coarse age bins for descriptive checks (not required by models but useful)
      - culture_cat: culture as categorical dtype
      - gender_cat: gender as categorical dtype

    Drops rows with missing required fields.
    """
    df = df.copy()

    # Required columns: y, age, gender, majority_first, culture
    required_cols = ['y', 'age', 'gender', 'majority_first', 'culture']
    df = df.dropna(subset=required_cols)

    # Binary: did the child choose one of the demonstrated options (majority or minority) vs the undemonstrated option
    df['ChoseDemonstrated'] = df['y'].apply(lambda x: 1 if x in [2, 3] else 0)

    # Among demonstrated choices, did the child choose the majority (1) or the minority (0). Undemonstrated choices become NaN.
    df['ChoseMajority'] = df['y'].map({2: 1, 3: 0, 1: np.nan})

    # Center age for interpretability and better numeric stability in interactions
    df['age_centered'] = df['age'] - df['age'].mean()

    # Create coarse age groups for descriptive exploration (4-6, 7-9, 10-12, 13-14)
    bins = [3, 6, 9, 12, 15]  # right edges; ages within 4..14 fall into these bins
    labels = ['4-6', '7-9', '10-12', '13-14']
    df['AgeGroup'] = pd.cut(df['age'], bins=bins, labels=labels, right=True, include_lowest=True)

    # Categorical versions of culture and gender for formula interfaces
    df['culture_cat'] = df['culture'].astype('category')
    df['gender_cat'] = df['gender'].astype('category')

    # Ensure majority_first is integer 0/1
    df['majority_first'] = df['majority_first'].astype(int)

    # Return the transformed dataframe with all columns required for modeling
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit two binomial (logistic) regression models to test whether reliance on social information and preference for the majority vary by age and culture.

    Model 1 (social reliance): ChoseDemonstrated ~ age_centered * culture + gender + majority_first
      - Outcome: ChoseDemonstrated (1 = used social information [majority or minority], 0 = chose undemonstrated)

    Model 2 (majority preference among demonstrated choices): ChoseMajority ~ age_centered * culture + gender + majority_first
      - Outcome: ChoseMajority (1 = chose majority, 0 = chose minority), restricted to rows where ChoseMajority is not NA

    Both models include an age-by-culture interaction to test whether developmental trajectories differ across cultures.

    Returns a dictionary with fitted model results (statsmodels GLMResults objects). The user can print summaries or extract coefficients.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    results = {}

    # Model 1: Reliance on social information (binary, using all rows)
    formula1 = 'ChoseDemonstrated ~ age_centered * C(culture_cat) + C(gender_cat) + majority_first'
    model1 = smf.glm(formula=formula1, data=df, family=sm.families.Binomial()).fit()
    results['model_demonstrated'] = model1

    # Model 2: Majority preference among those who used social information (drop undemonstrated choices)
    df_demo = df[df['ChoseMajority'].notnull()].copy()
    formula2 = 'ChoseMajority ~ age_centered * C(culture_cat) + C(gender_cat) + majority_first'
    model2 = smf.glm(formula=formula2, data=df_demo, family=sm.families.Binomial()).fit()
    results['model_majority'] = model2

    # Optionally print summaries (comment/uncomment as needed)
    # print('\n--- Reliance on social info (Model 1) ---')
    # print(model1.summary())
    # print('\n--- Majority preference among demonstrated choices (Model 2) ---')
    # print(model2.summary())

    return results


