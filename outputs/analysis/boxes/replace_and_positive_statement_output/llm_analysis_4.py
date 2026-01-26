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
    Transform the raw dataset to create the variables used in modeling.

    Produced columns (all included in returned df):
      - y: original categorical choice (int 1,2,3)
      - age: original age (years)
      - age_c: mean-centered age (age - mean(age))
      - culture: categorical site (dtype=category)
      - female: gender indicator (1 = girl, 0 = boy)
      - majority_first: original column coerced to int (0/1)
      - social_choice: binary (1 if chose a demonstrated option (majority or minority), 0 if chose the undemonstrated option)
      - is_demonstrated_choice: alias of social_choice for clarity
      - majority_choice: among demonstrated choices, 1 = majority, 0 = minority; NaN if the child chose the undemonstrated option
      - age_group: coarse developmental bins for descriptive checks (4-6,7-9,10-12,13-14)
    """
    df = df.copy()

    # Drop rows that lack essential data for the planned analyses
    df = df.dropna(subset=['y', 'age', 'culture', 'gender', 'majority_first'])

    # Ensure correct types for original variables
    df['y'] = df['y'].astype(int)
    df['age'] = df['age'].astype(float)
    df['culture'] = df['culture'].astype('category')
    df['gender'] = df['gender'].astype(int)
    df['majority_first'] = df['majority_first'].astype(int)

    # Derived variables
    # Center age to aid interpretability (main effects interpreted at mean age)
    df['age_c'] = df['age'] - df['age'].mean()

    # Binary gender indicator for 'female' (1 = girl, 0 = boy)
    df['female'] = (df['gender'] == 1).astype(int)

    # Did the child select one of the demonstrated options (majority or minority)?
    df['social_choice'] = df['y'].apply(lambda x: 1 if x in [2, 3] else 0).astype(int)
    df['is_demonstrated_choice'] = df['social_choice']

    # Among those who selected a demonstrated option, did they pick the majority? (1) or minority? (0)
    # For children who chose the undemonstrated option (y == 1) set NaN so this variable is only used with that subset
    def maj_choice_val(x):
        if x == 2:
            return 1.0
        elif x == 3:
            return 0.0
        else:
            return np.nan

    df['majority_choice'] = df['y'].apply(maj_choice_val).astype(float)

    # Coarse age groups for descriptive exploration (kept in dataframe but not required for main models)
    df['age_group'] = pd.cut(df['age'], bins=[3, 6, 9, 12, 15], labels=['4-6', '7-9', '10-12', '13-14'])

    # Keep and return the full dataframe with the new columns; downstream models will select appropriate subsets
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a small battery of models that directly answer whether (a) children's reliance on social information
    and (b) their preference for majority cues vary systematically with age and across cultures.

    Models fit:
      1) Logistic regression predicting social_choice (binary): social vs non-social choice.
         Formula: social_choice ~ age_c * C(culture) + female + majority_first
      2) Logistic regression predicting majority_choice among children who selected a demonstrated option.
         (Restricted to rows where is_demonstrated_choice == 1).
         Formula: majority_choice ~ age_c * C(culture) + female + majority_first
      3) Multinomial logistic regression predicting the full 3-category outcome (y = 1,2,3) to show the pattern
         across all choices jointly. We use Patsy to construct the design matrix and statsmodels MNLogit.

    The function returns a dict with the fitted model objects and prints summaries for quick inspection.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    import patsy

    results = {}

    # Model 1: Social reliance (binary)
    # Use logistic regression (statsmodels Logit via formula); suppress iteration output with disp=False
    model1 = smf.logit('social_choice ~ age_c * C(culture) + female + majority_first', data=df).fit(disp=False)
    print('\n=== Model 1: social_choice (logit) ===')
    print(model1.summary())
    results['model_social_reliance'] = model1

    # Model 2: Majority preference among those who used social information
    df_dem = df[df['is_demonstrated_choice'] == 1].copy()
    if df_dem.shape[0] < 10:
        print('\nWarning: too few demonstrated-choice rows for reliable estimation of majority preference model')
        model2 = None
    else:
        model2 = smf.logit('majority_choice ~ age_c * C(culture) + female + majority_first', data=df_dem).fit(disp=False)
        print('\n=== Model 2: majority_choice among demonstrated choices (logit) ===')
        print(model2.summary())
    results['model_majority_preference'] = model2

    # Model 3: Multinomial model for the full categorical outcome (y = 1,2,3)
    # Construct design matrix with patsy (this will expand C(culture) into dummies and include interaction terms)
    # For MNLogit we pass the integer-coded endogenous variable; here y is 1..3, so subtract 1 to get 0..2
    X = patsy.dmatrix('age_c * C(culture) + female + majority_first', data=df, return_type='dataframe')
    endog = (df['y'] - 1).astype(int)

    try:
        mn = sm.MNLogit(endog, X).fit(disp=False)
        print('\n=== Model 3: Multinomial logit for y (MNLogit) ===')
        print(mn.summary())
    except Exception as e:
        print('\nMultinomial model failed to converge or encountered an error:', e)
        mn = None
    results['model_multinomial_choice'] = mn

    return results


