from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/positive_leading_statement_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw Boston HMDA dataset into a cleaned dataframe with the exact columns used in the model.
    - Ensures binary columns are integers
    - Drops rows missing outcome, key IV or required controls
    - Standardizes continuous covariates (z-scores) to aid interpretation and numerical stability
    - Returns only the columns referenced in the model
    """
    df = df.copy()

    # Ensure columns exist
    expected_cols = ['female', 'accept', 'black', 'mortgage_credit', 'consumer_credit',
                     'PI_ratio', 'loan_to_value', 'bad_history', 'married', 'self_employed',
                     'housing_expense_ratio', 'denied_PMI']

    # Coerce expected columns to numeric where present
    for c in expected_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Binary columns to treat as integers
    bin_cols = ['female', 'accept', 'black', 'bad_history', 'married', 'self_employed', 'denied_PMI']

    # Continuous columns to standardize
    cont_cols = ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio']

    # Drop rows with missing outcome or primary IV or required control vars
    required = ['accept', 'female'] + bin_cols[1:] + cont_cols  # accept and female plus other required controls
    # Keep only required that actually exist in dataframe
    required_present = [c for c in required if c in df.columns]
    df = df.dropna(subset=required_present)

    # Convert binary columns to int (0/1)
    for c in bin_cols:
        if c in df.columns:
            # after dropna above, safe to cast
            df[c] = df[c].astype(int)

    # Standardize continuous columns (z-score). Use population std (ddof=0) for stability.
    for c in cont_cols:
        if c in df.columns:
            zname = c + '_z'
            std = df[c].std(ddof=0)
            if std == 0 or np.isnan(std):
                # avoid division by zero; produce zero-centered column
                df[zname] = df[c] - df[c].mean()
            else:
                df[zname] = (df[c] - df[c].mean()) / std

    # Final set of columns to return (these names must match the conceptual variables)
    keep_cols = [
        'female',
        'accept',
        'black',
        'mortgage_credit_z',
        'consumer_credit_z',
        'PI_ratio_z',
        'loan_to_value_z',
        'bad_history',
        'married',
        'self_employed',
        'housing_expense_ratio_z',
        'denied_PMI'
    ]

    # Keep only columns present in df (in case some original columns were missing) and reset index
    keep_present = [c for c in keep_cols if c in df.columns]
    df = df[keep_present].reset_index(drop=True)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a logistic regression predicting acceptance (accept) using applicant gender (female) and controls.
    Returns a dictionary with: full model summary (text), odds ratio for female with 95% CI, and average marginal effects summary.
    """
    import statsmodels.api as sm

    # Make a copy
    df = df.copy()

    # Define regressors (must match transformed column names)
    Xcols = [
        'female',
        'black',
        'mortgage_credit_z',
        'consumer_credit_z',
        'PI_ratio_z',
        'loan_to_value_z',
        'bad_history',
        'married',
        'self_employed',
        'housing_expense_ratio_z',
        'denied_PMI'
    ]

    # Keep only columns present in the passed df
    Xcols_present = [c for c in Xcols if c in df.columns]
    if 'accept' not in df.columns:
        raise ValueError("Dependent variable 'accept' not found in dataframe passed to model(). Make sure you ran transform().")

    X = df[Xcols_present]
    X = sm.add_constant(X)
    y = df['accept']

    # Fit logistic regression (maximum likelihood)
    logit = sm.Logit(y, X)
    res = logit.fit(disp=False)

    # Odds ratio for female (if present) and 95% CI
    if 'female' in res.params.index:
        or_female = float(np.exp(res.params['female']))
        ci = res.conf_int().loc['female']
        ci_or = np.exp(ci)
        or_ci_lower = float(ci_or.iloc[0])
        or_ci_upper = float(ci_or.iloc[1])
    else:
        or_female = None
        or_ci_lower = None
        or_ci_upper = None

    # Average marginal effects (marginal effect of female on probability of acceptance)
    try:
        marg = res.get_margeff(at='overall', method='dydx')
        marg_text = marg.summary().as_text()
    except Exception as e:
        marg_text = f"Failed to compute marginal effects: {e}"

    results = {
        'model_summary': res.summary().as_text(),
        'odds_ratio_female': or_female,
        'odds_ratio_female_ci_lower': or_ci_lower,
        'odds_ratio_female_ci_upper': or_ci_upper,
        'marginal_effects_text': marg_text,
        'statsmodels_result_object': res  # returned for advanced inspection if caller wishes
    }

    # Print key outputs for interactive use
    print(res.summary())
    if or_female is not None:
        print('\nOdds ratio for female =', or_female, '(95% CI =', (or_ci_lower, or_ci_upper), ')')
    print('\nAverage marginal effects (summary):\n', marg_text)

    return results


