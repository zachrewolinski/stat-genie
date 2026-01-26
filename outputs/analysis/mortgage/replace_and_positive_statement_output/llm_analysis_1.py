from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/replace_and_positive_statement_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw Boston Fed mortgage dataset to a modeling-ready dataframe.

    Steps:
    - Make a copy of the input.
    - Drop rows missing the outcome ('accept') or the main IV ('female').
    - Coerce numeric columns, impute medians for continuous controls.
    - Impute zeros for binary controls and cast to int.
    - Create standardized (z-scored) versions of continuous controls for stable coefficient scaling.
    - Create an interaction term female_black = female * black to test whether the effect of gender differs by race.

    Returns the transformed dataframe containing the exact columns used in the model.
    """
    df = df.copy()

    # Ensure key columns exist
    required = ['accept', 'female']
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Required column missing: {col}")

    # Drop rows missing DV or main IV
    df = df.dropna(subset=['accept', 'female']).copy()

    # Continuous columns to coerce + standardize
    cont_cols = ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio']
    for c in cont_cols:
        if c not in df.columns:
            # create NaN column if missing to avoid crashes; will be filled with 0 median later
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors='coerce')
        med = df[c].median()
        # If median is NaN (all missing), fill with 0
        if np.isnan(med):
            med = 0.0
        df[c] = df[c].fillna(med)
        mean = df[c].mean()
        std = df[c].std(ddof=0)
        if std == 0 or np.isnan(std):
            std = 1.0
        df[c + '_z'] = (df[c] - mean) / std

    # Binary controls to coerce and impute
    bin_cols = ['black', 'bad_history', 'self_employed', 'married', 'female']
    for c in bin_cols:
        if c not in df.columns:
            df[c] = 0
        # coerce to numeric then to 0/1 int
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        # Some columns might be non-binary (e.g., 0/1 floats); convert to 0/1 by rounding then cast
        df[c] = df[c].round().astype(int)
        # ensure values are 0/1
        df[c] = df[c].apply(lambda x: 1 if x >= 1 else 0)

    # Ensure accept is binary 0/1 and int
    df['accept'] = pd.to_numeric(df['accept'], errors='coerce').fillna(0).round().astype(int)
    df['accept'] = df['accept'].apply(lambda x: 1 if x >= 1 else 0)

    # Create interaction term for intersectional test
    df['female_black'] = df['female'] * df['black']

    # Final check: keep only rows with non-missing values for model columns
    model_cols = ['accept', 'female', 'black', 'female_black', 'mortgage_credit_z', 'consumer_credit_z',
                  'PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z', 'bad_history', 'self_employed', 'married']
    df = df.copy()
    # if any of model_cols missing (shouldn't be), raise
    for c in model_cols:
        if c not in df.columns:
            raise KeyError(f"Expected column missing after transform: {c}")

    # Return dataframe with model columns plus original columns (safe) but model will only use model_cols
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression to estimate the effect of gender (female) on mortgage acceptance,
    controlling for creditworthiness and other applicant attributes. Also test the interaction
    between female and Black applicant (female_black).

    Returns a dictionary with the fitted model object, odds ratios table, and estimated marginal
    effect of being female on acceptance probability (average marginal effect).
    """
    # Ensure transformed dataframe has the required columns
    cols_needed = ['accept', 'female', 'black', 'female_black', 'mortgage_credit_z', 'consumer_credit_z',
                   'PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z', 'bad_history', 'self_employed', 'married']
    for c in cols_needed:
        if c not in df.columns:
            raise KeyError(f"Required column for model is missing: {c}")

    # Define model matrix
    X_cols = ['female', 'black', 'female_black', 'mortgage_credit_z', 'consumer_credit_z',
              'PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z', 'bad_history', 'self_employed', 'married']
    X = df[X_cols].astype(float)
    X = sm.add_constant(X, has_constant='add')
    y = df['accept'].astype(float)

    # Fit logistic regression (maximum likelihood)
    logit = sm.Logit(y, X)
    try:
        result = logit.fit(disp=False)
    except Exception as e:
        # If Logit has problems (e.g., perfect separation), try GLM with binomial family
        glm = sm.GLM(y, X, family=sm.families.Binomial())
        result = glm.fit()

    # Compute odds ratios and 95% CI
    params = result.params
    try:
        conf = result.conf_int()
        conf.columns = ['2.5%', '97.5%']
        odds_ratios = pd.DataFrame({
            'OR': np.exp(params),
            '2.5%': np.exp(conf['2.5%']),
            '97.5%': np.exp(conf['97.5%'])
        })
    except Exception:
        # fallback if no conf_int available
        odds_ratios = pd.DataFrame({'OR': np.exp(params)})

    # Average marginal effect of being female on probability of acceptance
    marg_eff = None
    try:
        # Statsmodels discrete fit objects support get_margeff for Logit; if result is GLM this may fail
        me = result.get_margeff(at='overall', method='dydx')
        marg_eff = me.summary_frame()
    except Exception as e:
        marg_eff = str(e)

    # Print a brief summary for interactive use
    try:
        print(result.summary())
    except Exception:
        pass

    # Return objects for programmatic inspection
    return {
        'model_result': result,
        'odds_ratios': odds_ratios,
        'marginal_effect_female': marg_eff,
        'X_columns': X_cols
    }


