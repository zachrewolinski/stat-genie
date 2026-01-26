from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/negative_leading_statement_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the modeling dataframe. The function:
    - drops rows with missing values in the key columns
    - creates the multinomial outcome y_code (0,1,2)
    - creates a mean-centered age variable age_c
    - encodes gender as gender_female (1=girl, 0=boy)
    - ensures majority_first is binary
    - creates explicit culture dummy columns culture_2..culture_8 (culture_1 is reference)
    - creates interaction terms age_c_culture_2..age_c_culture_8 to test whether developmental change differs by culture

    Returns a dataframe containing exactly the columns used in the model.
    """
    df = df.copy()

    # Drop rows with missing values in any variables we will use
    needed_cols = ['y', 'age', 'gender', 'majority_first', 'culture']
    df = df.dropna(subset=needed_cols)

    # Ensure integer types where appropriate
    df['y'] = df['y'].astype(int)
    df['age'] = df['age'].astype(float)
    df['gender'] = df['gender'].astype(int)
    df['majority_first'] = df['majority_first'].astype(int)
    df['culture'] = df['culture'].astype(int)

    # Dependent variable: map original y (1,2,3) -> y_code (0,1,2)
    # 1 = unchosen option -> 0 ; 2 = majority -> 1 ; 3 = minority -> 2
    df['y_code'] = df['y'] - 1

    # Center age for interpretability and to reduce collinearity with interactions
    df['age_c'] = df['age'] - df['age'].mean()

    # Gender: create female indicator (1 = girl, 0 = boy)
    # Original coding: 1 = girl, 2 = boy
    df['gender_female'] = (df['gender'] == 1).astype(int)

    # Ensure majority_first is 0/1
    df['majority_first'] = df['majority_first'].astype(int)

    # Create explicit culture dummy columns for cultures 2..8 (culture 1 is reference)
    # This guarantees the same column names irrespective of which cultures appear in the sample
    for i in range(2, 9):
        col = f'culture_{i}'
        df[col] = (df['culture'] == i).astype(int)

    # Create age-by-culture interaction terms for cultures 2..8
    for i in range(2, 9):
        dcol = f'culture_{i}'
        icol = f'age_c_culture_{i}'
        df[icol] = df['age_c'] * df[dcol]

    # Select and return only the columns needed for modeling
    model_cols = [
        'y_code',
        'age', 'age_c',
        'gender_female',
        'majority_first',
        'culture_2', 'culture_3', 'culture_4', 'culture_5', 'culture_6', 'culture_7', 'culture_8',
        'age_c_culture_2', 'age_c_culture_3', 'age_c_culture_4', 'age_c_culture_5', 'age_c_culture_6', 'age_c_culture_7', 'age_c_culture_8'
    ]

    # Some rows may have been dropped earlier; ensure all model columns exist (they should) and return subset
    return df[model_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a multinomial logistic regression to test whether children's choices (y_code) depend on age,
    culture, and their interaction -- controlling for gender and majority_first.

    Performs likelihood-ratio tests for:
      1) Main effect of age (developmental change) -- test by removing age and all age-by-culture interactions
      2) Main effect of culture -- test by removing culture dummies and the interactions
      3) Age x Culture interactions -- test by removing interaction terms only

    Returns a dictionary with the fitted full model and LR-test results.
    """
    import statsmodels.api as sm
    from scipy import stats

    df = df.copy()

    # Define predictor column names exactly as created in transform
    culture_cols = ['culture_2', 'culture_3', 'culture_4', 'culture_5', 'culture_6', 'culture_7', 'culture_8']
    interaction_cols = [f'age_c_culture_{i}' for i in range(2, 9)]
    base_predictors = ['age_c', 'gender_female', 'majority_first']

    predictors = base_predictors + culture_cols + interaction_cols

    # Exog with constant
    exog = sm.add_constant(df[predictors], has_constant='add')
    endog = df['y_code'].astype(int)

    # Fit full multinomial logistic model (reference category will be the first outcome 0)
    try:
        full_model = sm.MNLogit(endog, exog).fit(method='newton', maxiter=200, disp=False)
    except Exception as e:
        # fallback: try a different solver if convergence issues occur
        full_model = sm.MNLogit(endog, exog).fit(method='bfgs', maxiter=200, disp=False)

    # Helper to fit reduced model given columns to keep in exog (columns should include constant if desired)
    def fit_reduced(keep_cols):
        exog_r = sm.add_constant(df[keep_cols], has_constant='add')
        try:
            m = sm.MNLogit(endog, exog_r).fit(method='newton', maxiter=200, disp=False)
        except Exception:
            m = sm.MNLogit(endog, exog_r).fit(method='bfgs', maxiter=200, disp=False)
        return m

    results = {}

    # 1) Test main effect of age (remove age_c and all age-by-culture interactions)
    keep_cols_age_test = [c for c in predictors if c not in (['age_c'] + interaction_cols)]
    reduced_age = fit_reduced(keep_cols_age_test)
    lr_stat_age = 2 * (full_model.llf - reduced_age.llf)
    df_age = full_model.params.size - reduced_age.params.size
    pval_age = stats.chi2.sf(lr_stat_age, df_age)
    results['age_main_effect'] = {
        'lr_stat': float(lr_stat_age),
        'df': int(df_age),
        'pvalue': float(pval_age),
        'full_llf': float(full_model.llf),
        'reduced_llf': float(reduced_age.llf)
    }

    # 2) Test main effect of culture (remove culture dummies and interactions)
    keep_cols_culture_test = [c for c in predictors if c not in (culture_cols + interaction_cols)]
    reduced_culture = fit_reduced(keep_cols_culture_test)
    lr_stat_culture = 2 * (full_model.llf - reduced_culture.llf)
    df_culture = full_model.params.size - reduced_culture.params.size
    pval_culture = stats.chi2.sf(lr_stat_culture, df_culture)
    results['culture_main_effect'] = {
        'lr_stat': float(lr_stat_culture),
        'df': int(df_culture),
        'pvalue': float(pval_culture),
        'full_llf': float(full_model.llf),
        'reduced_llf': float(reduced_culture.llf)
    }

    # 3) Test age x culture interactions (remove interaction terms only)
    keep_cols_interaction_test = [c for c in predictors if c not in interaction_cols]
    reduced_interaction = fit_reduced(keep_cols_interaction_test)
    lr_stat_inter = 2 * (full_model.llf - reduced_interaction.llf)
    df_inter = full_model.params.size - reduced_interaction.params.size
    pval_inter = stats.chi2.sf(lr_stat_inter, df_inter)
    results['age_by_culture_interaction'] = {
        'lr_stat': float(lr_stat_inter),
        'df': int(df_inter),
        'pvalue': float(pval_inter),
        'full_llf': float(full_model.llf),
        'reduced_llf': float(reduced_interaction.llf)
    }

    # Add parameter estimates and a short summary
    results['full_model_params'] = full_model.params.to_dict()
    results['full_model_summary'] = full_model.summary().as_text()

    return results


