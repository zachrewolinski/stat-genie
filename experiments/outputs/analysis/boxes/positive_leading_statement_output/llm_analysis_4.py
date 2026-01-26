from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/positive_leading_statement_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe with all variables used in modeling.

    Produces these columns (in addition to keeping original columns where relevant):
      - age_c: age mean-centered
      - age_group: categorical developmental bins (4-6,7-9,10-12,13-14)
      - gender_f: 1 if girl (original gender==1), 0 if boy (original gender==2)
      - IsMajority: 1 if y==2 (child chose majority), 0 otherwise
      - IsDemonstrated: 1 if y in {2,3} (chose either demonstrated option), 0 otherwise
      - culture: integer culture/site id (kept from original but cast to int)

    Rows with missing critical fields (y, age, culture) are dropped.
    """
    df = df.copy()

    # Drop rows missing critical variables
    df = df.dropna(subset=['y', 'age', 'culture'])

    # Ensure types
    df['y'] = df['y'].astype(int)
    df['age'] = df['age'].astype(float)
    # mean-centered age for numerical stability and interpretation of interactions
    df['age_c'] = df['age'] - df['age'].mean()

    # Gender: dataset uses 1=girl, 2=boy. Create female indicator: 1 = girl, 0 = boy
    df['gender_f'] = df['gender'].map({1: 1, 2: 0})
    df['gender_f'] = df['gender_f'].fillna(0).astype(int)

    # Binary dependent variables for targeted analyses
    df['IsMajority'] = (df['y'] == 2).astype(int)
    df['IsDemonstrated'] = df['y'].isin([2, 3]).astype(int)

    # Age groups (developmental stages) for descriptive checks and non-linear effects
    bins = [3, 6, 9, 12, 14.1]  # right edge slightly above max to include 14
    labels = ['4-6', '7-9', '10-12', '13-14']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=True)

    # Ensure culture is integer-coded (kept as is but ensure type)
    df['culture'] = df['culture'].astype(int)

    # Ensure majority_first is binary
    if 'majority_first' in df.columns:
        df['majority_first'] = df['majority_first'].astype(int)
    else:
        # If not present, create a column of zeros (defensive)
        df['majority_first'] = 0

    # Return only the columns needed for modeling plus originals for traceability
    keep_cols = list(df.columns)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit three complementary models to answer whether reliance on social information and preference for majority cues vary across cultures and development:

    1) Multinomial logistic regression predicting the 3-way choice (y = 1 undemonstrated, 2 majority, 3 minority) from age (centered), culture dummies, gender, majority_first and culture dummies. This tests whether the distribution over the three choices depends on age and culture.

    2) Binary logistic regression predicting IsMajority (chose majority vs not) with an age-by-culture interaction to specifically test developmental and cultural variation in majority preference.

    3) Binary logistic regression predicting IsDemonstrated (chose any demonstrated option vs undemonstrated) with the same interaction to test overall reliance on social information across age and cultures.

    Returns a dict with fitted model results objects (statsmodels). These objects provide coefficients, standard errors, p-values, and summary tables.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.discrete.discrete_model import MNLogit
    results = {}

    # 1) Multinomial logistic regression (y has categories 1,2,3)
    # Build exogenous matrix with culture dummies (drop first to avoid collinearity)
    culture_dummies = pd.get_dummies(df['culture'].astype(str), prefix='culture', drop_first=True)
    exog = pd.concat([df[['age_c', 'gender_f', 'majority_first']].reset_index(drop=True), culture_dummies.reset_index(drop=True)], axis=1)
    exog = sm.add_constant(exog, has_constant='add')
    endog = df['y'].astype(int)

    # Fit MNLogit (statsmodels). Use try/except to catch potential convergence issues.
    try:
        mn_model = MNLogit(endog, exog)
        mn_res = mn_model.fit(method='newton', maxiter=200, disp=False)
    except Exception:
        # Retry with a different solver or smaller tolerance if necessary
        mn_res = MNLogit(endog, exog).fit(method='bfgs', maxiter=200, disp=False)

    results['mnlogit'] = mn_res

    # 2) Logistic regression for majority preference with age-by-culture interaction
    # Use formula API so we can specify C(culture) and interaction easily
    formula_majority = 'IsMajority ~ age_c * C(culture) + gender_f + majority_first'
    logit_maj = smf.logit(formula_majority, data=df).fit(disp=False)
    results['logit_majority'] = logit_maj

    # 3) Logistic regression for demonstrated vs undemonstrated (reliance on social information)
    formula_demo = 'IsDemonstrated ~ age_c * C(culture) + gender_f + majority_first'
    logit_demo = smf.logit(formula_demo, data=df).fit(disp=False)
    results['logit_demonstrated'] = logit_demo

    # Return the fitted model result objects for inspection
    return results


