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
    Transform the raw dataset into a dataframe ready for modeling.

    Produces the following new/derived columns used in the models:
      - SocialInfoUsed: binary (1 = chose demonstrated option (majority/minority), 0 = chose undemonstrated option)
      - MajorityChoice: 1 if majority chosen, 0 if minority chosen, NaN if undemonstrated choice
      - age_c: mean-centered age for more stable interpretation of interactions
      - AgeGroup: categorical age bins (early/middle/late childhood)
      - culture_cat: categorical site label (string) for use with formula-based modeling
      - gender_cat: 'girl'/'boy' categorical label

    The function also drops rows missing the key variables (y, age, culture) required for the analyses.
    """

    df = df.copy()

    # Keep only rows with the core variables present
    df = df.dropna(subset=['y', 'age', 'culture'])

    # Ensure numeric types where expected
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['culture'] = pd.to_numeric(df['culture'], errors='coerce')
    df['majority_first'] = pd.to_numeric(df['majority_first'], errors='coerce')
    df['gender'] = pd.to_numeric(df['gender'], errors='coerce')

    # Dependent variables
    # SocialInfoUsed: 1 if child chose a demonstrated option (majority y==2 or minority y==3), else 0
    df['SocialInfoUsed'] = df['y'].isin([2, 3]).astype(float)

    # MajorityChoice: among demonstrated choices, 1 if majority (y==2), 0 if minority (y==3), NaN otherwise
    df['MajorityChoice'] = df['y'].map({2: 1.0, 3: 0.0})

    # Keep original y for reference
    df['y'] = df['y'].astype(int)

    # Age transformations
    df['age'] = df['age'].astype(float)
    df['age_c'] = df['age'] - df['age'].mean()

    # AgeGroup: coarse developmental stages for descriptive checks and possible subgroup analyses
    # bins: 4-6 (early childhood), 7-9 (middle childhood), 10-14 (late childhood)
    bins = [3.5, 6.5, 9.5, 14.5]
    labels = ['early', 'middle', 'late']
    df['AgeGroup'] = pd.cut(df['age'], bins=bins, labels=labels)

    # Culture as categorical string for use in formulas (C(culture_cat))
    # Keep original numeric culture for reference
    df['culture_cat'] = 'C' + df['culture'].astype(int).astype(str)

    # Gender as categorical label
    # Original coding: 1 = girl, 2 = boy
    df['gender_cat'] = df['gender'].map({1: 'girl', 2: 'boy'})

    # Sanity: ensure majority_first is 0/1, but allow missing
    df['majority_first'] = df['majority_first'].apply(lambda x: np.nan if pd.isnull(x) else int(x))

    # Return only columns necessary for modeling + original y/covariates for clarity
    out_cols = [
        'y',
        'SocialInfoUsed',
        'MajorityChoice',
        'age',
        'age_c',
        'AgeGroup',
        'culture',
        'culture_cat',
        'gender',
        'gender_cat',
        'majority_first'
    ]
    return df[out_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit two complementary models to answer whether reliance on social information and preference for majority cues vary across cultures and developmental stages.

    Models:
      1) social_model: logistic regression predicting SocialInfoUsed (1 = chose a demonstrated option) using age (centered), culture (factor), their interaction, and controls (gender, majority_first).
      2) majority_model: logistic regression among those who chose a demonstrated option (MajorityChoice not null) predicting whether the child chose the majority (1) vs minority (0) using the same predictors.

    Returns a dict with the fitted results objects for further inspection.
    """

    import statsmodels.formula.api as smf

    results = {}

    # Model 1: Reliance on social information (binary)
    df1 = df.dropna(subset=['SocialInfoUsed', 'age_c', 'culture_cat', 'gender_cat'])

    formula_social = 'SocialInfoUsed ~ age_c + C(culture_cat) + age_c:C(culture_cat) + C(gender_cat) + majority_first'
    try:
        social_model = smf.logit(formula=formula_social, data=df1).fit(disp=False)
        results['social_model'] = social_model
    except Exception as e:
        # Provide a fallback (GLM with binomial family) if logit fails
        try:
            social_model_glm = smf.glm(formula=formula_social, data=df1, family=smf.families.Binomial()).fit()
            results['social_model'] = social_model_glm
            results['social_model_warning'] = f'Logit failed, used GLM Binomial: {e}'
        except Exception as e2:
            results['social_model_error'] = str(e2)

    # Model 2: Preference for majority cues among those who used social information
    df2 = df[df['MajorityChoice'].notnull()].copy()
    df2 = df2.dropna(subset=['MajorityChoice', 'age_c', 'culture_cat', 'gender_cat'])

    formula_majority = 'MajorityChoice ~ age_c + C(culture_cat) + age_c:C(culture_cat) + C(gender_cat) + majority_first'
    try:
        majority_model = smf.logit(formula=formula_majority, data=df2).fit(disp=False)
        results['majority_model'] = majority_model
    except Exception as e:
        # Fallback to GLM binomial
        try:
            majority_model_glm = smf.glm(formula=formula_majority, data=df2, family=smf.families.Binomial()).fit()
            results['majority_model'] = majority_model_glm
            results['majority_model_warning'] = f'Logit failed, used GLM Binomial: {e}'
        except Exception as e2:
            results['majority_model_error'] = str(e2)

    # Recommended next steps for inference (not executed here):
    # - Examine coefficients for age_c and the age_c:C(culture_cat) interaction terms to assess whether age-related changes differ by culture.
    # - Use Wald tests or likelihood-ratio tests to assess significance of interaction blocks (e.g., compare full model to model without age*Culture interactions).
    # - Plot predicted probabilities across age for each culture to visualize developmental trajectories.

    return results


