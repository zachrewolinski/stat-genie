from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/noperturb_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the original dataframe into the final dataframe used by the models.

    Produces:
    - y (int) : original outcome (1,2,3) kept as-is
    - social_choice (0/1) : 1 if y in {2,3} (chose demonstrated option), else 0
    - majority_choice (0/1) : 1 if y == 2 else 0
    - is_boy (0/1) : 1 if gender == 2, 0 if gender == 1
    - majority_first (0/1) : ensures binary dtype
    - age_centered : age minus mean(age)
    - culture_2..culture_8 : dummy columns for culture with culture_1 as reference (drop_first)
    - age_x_culture_* : interaction columns between age_centered and each culture dummy
    """

    # Work on a copy
    df = df.copy()

    # Drop rows missing essential variables
    essential_cols = ['y', 'age', 'gender', 'culture', 'majority_first']
    df = df.dropna(subset=essential_cols)

    # Ensure correct dtypes
    df['y'] = df['y'].astype(int)
    df['age'] = df['age'].astype(float)
    df['gender'] = df['gender'].astype(int)
    # majority_first should be 0/1 already; coerce to int
    df['majority_first'] = df['majority_first'].astype(int)

    # Derived binary variables
    df['social_choice'] = df['y'].isin([2, 3]).astype(int)
    df['majority_choice'] = (df['y'] == 2).astype(int)

    # Gender binary control: dataset encoding 1 = girl, 2 = boy
    df['is_boy'] = (df['gender'] == 2).astype(int)

    # Center age for interpretability / interactions
    df['age_centered'] = df['age'] - df['age'].mean()

    # Culture dummies: create k-1 dummies. We expect culture IDs from 1..8; drop_first -> drop culture_1 (reference)
    # Convert to string first to avoid creating numeric-suffixed columns in unexpected ways
    df['culture'] = df['culture'].astype(int)
    culture_dummies = pd.get_dummies(df['culture'], prefix='culture', drop_first=True)
    df = pd.concat([df, culture_dummies], axis=1)

    # Compute interaction terms between age_centered and culture dummies (moderation terms)
    culture_cols = [c for c in df.columns if c.startswith('culture_')]
    for c in culture_cols:
        inter_name = f'age_x_{c}'
        df[inter_name] = df['age_centered'] * df[c]

    # Final check: ensure expected dummy columns exist even if some cultures are missing in the sample
    # (this makes model code robust). If missing, create columns filled with zeros for 2..8.
    expected_culture_cols = [f'culture_{i}' for i in range(2, 9)]
    for col in expected_culture_cols:
        if col not in df.columns:
            df[col] = 0
            df[col] = df[col].astype(int)
            # also create corresponding interaction column
            inter_col = f'age_x_{col}'
            df[inter_col] = 0.0

    # Re-order columns minimally (not required but helpful)
    # Return final dataframe
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Run statistical models to answer the research question.

    Models fitted:
    - Binary logistic regression predicting social_choice (reliance on social information) with predictors:
      age_centered, is_boy, majority_first, culture dummies (culture_2..culture_8), and interactions age_x_culture_*

    - Binary logistic regression predicting majority_choice among trials where a demonstrated option was chosen (social_choice == 1).
      Same predictors as above; models preference for majority vs minority among social choices.

    - Multinomial logistic regression (MNLogit) predicting the original 3-level outcome y (0/1/2 after re-coding) using main effects
      (age_centered, is_boy, majority_first, culture dummies). Interactions are omitted in the MNLogit here to aid convergence,
      but could be included if desired and sample size supports it.

    Returns a dict containing fitted results objects:
    { 'social_choice_model': result_obj, 'majority_choice_model': result_obj_or_None, 'mnlogit_model': result_obj }
    """

    results = {}

    # Build list of culture dummy columns (expected names produced by transform)
    culture_dummies = [f'culture_{i}' for i in range(2, 9)]
    interaction_cols = [f'age_x_culture_{i}' for i in range(2, 9)]

    # Base predictors
    base_predictors = ['age_centered', 'is_boy', 'majority_first']

    # For logistic models include culture dummies and their interactions
    logit_predictors = base_predictors + culture_dummies + interaction_cols

    # Ensure predictors exist in df (transform should have created them). If any missing, fill with zeros.
    for col in logit_predictors:
        if col not in df.columns:
            df[col] = 0

    # Add constant
    exog = sm.add_constant(df[logit_predictors], has_constant='add')

    # 1) Logistic model: social_choice ~ predictors
    try:
        mod_social = sm.Logit(df['social_choice'], exog)
        res_social = mod_social.fit(disp=False)
        # Robust covariance (HC3) for inference
        res_social_robust = res_social.get_robustcov_results(cov_type='HC3')
        results['social_choice_model'] = res_social_robust
    except Exception as e:
        # If model fails (e.g., separability), store the exception for diagnostics
        results['social_choice_model'] = {'error': str(e)}

    # 2) Logistic model among social choices: majority_choice ~ predictors
    df_social = df[df['social_choice'] == 1].copy()
    if df_social.shape[0] < 10:
        # Not enough data to fit a reliable model
        results['majority_choice_model'] = {'error': f'Not enough social-choice cases to fit model (n={df_social.shape[0]})'}
    else:
        # Build exog for subset (ensuring same columns)
        for col in logit_predictors:
            if col not in df_social.columns:
                df_social[col] = 0
        exog_soc = sm.add_constant(df_social[logit_predictors], has_constant='add')
        try:
            mod_major = sm.Logit(df_social['majority_choice'], exog_soc)
            res_major = mod_major.fit(disp=False)
            res_major_robust = res_major.get_robustcov_results(cov_type='HC3')
            results['majority_choice_model'] = res_major_robust
        except Exception as e:
            results['majority_choice_model'] = {'error': str(e)}

    # 3) Multinomial model on the original 3-level outcome y using main effects (no interactions to aid convergence)
    # Re-code y to 0..K-1
    try:
        y_mn = (df['y'] - df['y'].min()).astype(int)
        mn_predictors = base_predictors + culture_dummies
        for col in mn_predictors:
            if col not in df.columns:
                df[col] = 0
        exog_mn = sm.add_constant(df[mn_predictors], has_constant='add')
        mod_mn = sm.MNLogit(y_mn, exog_mn)
        res_mn = mod_mn.fit(disp=False, maxiter=200)
        results['mnlogit_model'] = res_mn
    except Exception as e:
        results['mnlogit_model'] = {'error': str(e)}

    return results


