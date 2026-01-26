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
    Clean and prepare the Boston mortgage dataset for modeling.

    Produces a dataframe with the binary dependent variable 'accept' and all
essential independent/control columns named in the conceptual variables.
    """
    # Work on a copy
    df = df.copy()

    # Drop index-like unnamed column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Ensure expected columns exist
    expected_cols = [
        'female', 'accept', 'black', 'mortgage_credit', 'consumer_credit',
        'bad_history', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio',
        'married', 'self_employed'
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe missing required columns: {missing}")

    # Convert binary columns to integers (0/1)
    for col in ['female', 'accept', 'black', 'bad_history', 'married', 'self_employed']:
        # coerce to numeric then to int (where appropriate)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # If values are floats but really 0/1, round and convert
        df[col] = df[col].round(0).astype('Int64')

    # Convert numeric controls
    for col in ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove rows with missing key variables (DV, IV, and controls)
    required_for_model = expected_cols
    df = df.dropna(subset=required_for_model).reset_index(drop=True)

    # Sanity checks and simple winsorization for extreme continuous ratios
    # Cap unrealistic PI_ratio at [0, 2] and loan_to_value at [0, 2]
    df['PI_ratio'] = df['PI_ratio'].clip(lower=0.0, upper=2.0)
    df['loan_to_value'] = df['loan_to_value'].clip(lower=0.0, upper=2.0)
    df['housing_expense_ratio'] = df['housing_expense_ratio'].clip(lower=0.0, upper=2.0)

    # Ensure integer dtype for final binary columns
    df['female'] = df['female'].astype(int)
    df['accept'] = df['accept'].astype(int)
    df['black'] = df['black'].astype(int)
    df['bad_history'] = df['bad_history'].astype(int)
    df['married'] = df['married'].astype(int)
    df['self_employed'] = df['self_employed'].astype(int)

    # (Optional) create an interaction term female * black for exploratory intersectionality
    df['female_black_interaction'] = df['female'] * df['black']

    # Return dataframe with exactly the columns needed by the model (plus interaction)
    keep_cols = [
        'accept', 'female', 'black', 'mortgage_credit', 'consumer_credit',
        'bad_history', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio',
        'married', 'self_employed', 'female_black_interaction'
    ]
    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a logistic regression (GLM binomial) predicting acceptance of mortgage
    application from applicant gender and a set of controls.

    Returns a dictionary with the fitted model object, odds ratios, confidence
    intervals, p-values, number of observations, McFadden pseudo-R^2, and an
    average marginal effect for the female indicator.
    """
    results = {}

    # Define model variables (match the transformed dataframe columns)
    y_col = 'accept'
    model_vars = [
        'female', 'black', 'mortgage_credit', 'consumer_credit', 'bad_history',
        'PI_ratio', 'loan_to_value', 'housing_expense_ratio', 'married', 'self_employed'
    ]

    # Safety: confirm columns present
    missing = [c for c in [y_col] + model_vars if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for modeling: {missing}")

    # Prepare design matrix with constant
    X = sm.add_constant(df[model_vars], has_constant='add')
    y = df[y_col]

    # Fit a binomial GLM with robust (HC3) standard errors
    glm_binom = sm.GLM(y, X, family=sm.families.Binomial())
    res = glm_binom.fit(cov_type='HC3')

    # Odds ratios and confidence intervals
    params = res.params
    conf = res.conf_int()
    conf.columns = ['ci_lower', 'ci_upper']
    odds_ratios = np.exp(params)
    or_conf = np.exp(conf)

    # p-values and nobs
    pvalues = res.pvalues
    nobs = int(res.nobs)

    # McFadden pseudo R^2: 1 - (ll_model / ll_null)
    # Fit intercept-only (null) model to get llnull
    X_null = sm.add_constant(pd.DataFrame({'intercept': np.ones(len(y))}), has_constant='add')
    null_mod = sm.GLM(y, X_null, family=sm.families.Binomial())
    null_res = null_mod.fit()
    llf = res.llf
    llnull = null_res.llf
    pseudo_r2 = 1.0 - (llf / llnull) if llnull != 0 else np.nan

    # Average marginal effect of female (discrete change): average predicted prob when female=1 minus female=0
    X_f1 = X.copy()
    X_f0 = X.copy()
    X_f1['female'] = 1
    X_f0['female'] = 0
    pred_f1 = res.predict(X_f1)
    pred_f0 = res.predict(X_f0)
    avg_marginal_effect_female = float(np.mean(pred_f1 - pred_f0))

    # Package results
    results['model_result'] = res
    # Convert series/dataframes to plain pandas structures for easier downstream use
    results['odds_ratios'] = odds_ratios
    results['or_conf_int'] = or_conf
    results['pvalues'] = pvalues
    results['nobs'] = nobs
    results['mcfadden_pseudo_r2'] = pseudo_r2
    results['avg_marginal_effect_female'] = avg_marginal_effect_female

    # Also include a short textual summary for convenience
    summary_text = (
        f"Logistic GLM (Binomial) fitted with {nobs} observations.\n"
        f"Key coefficient for 'female': coef={params.get('female'):.4f}, "
        f"OR={odds_ratios.get('female'):.3f}, p={pvalues.get('female'):.4f}.\n"
        f"Average marginal effect of female on approval probability = {avg_marginal_effect_female:.4f} (probability points).\n"
        f"McFadden pseudo-R^2 = {pseudo_r2:.4f}.\n"
    )
    results['summary_text'] = summary_text

    return results


