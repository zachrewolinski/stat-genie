from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/negative_leading_statement_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataset for modeling.
    - Select relevant columns
    - Drop rows with missing values on variables used in the model
    - Ensure binary indicators are integers (0/1)
    - Create standardized (z-scored) versions of continuous predictors used as controls
    - Return a dataframe that contains the outcome, the focal IV, and all control columns named exactly as in the conceptual variables
    """
    df = df.copy()

    # Columns needed for analysis
    needed = [
        'female',
        'accept',
        'black',
        'housing_expense_ratio',
        'self_employed',
        'married',
        'mortgage_credit',
        'consumer_credit',
        'bad_history',
        'PI_ratio',
        'loan_to_value',
        'denied_PMI'
    ]

    # Keep only needed columns (if any are missing in the provided df this will raise KeyError)
    df = df[needed]

    # Drop rows with missing values in any of the variables required for the model
    df = df.dropna()

    # Ensure binary columns are integer 0/1
    binary_cols = ['female', 'accept', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI']
    for c in binary_cols:
        # Some datasets may have floats like 0.0/1.0; cast to int
        df[c] = df[c].astype(int)

    # Standardize continuous controls (z-scores). Use population std (ddof=0) for stable scaling.
    cont_cols = ['housing_expense_ratio', 'PI_ratio', 'loan_to_value', 'mortgage_credit', 'consumer_credit']
    for c in cont_cols:
        mean = df[c].mean()
        std = df[c].std(ddof=0)
        # If zero variance (unlikely here), set z to 0 to avoid division by zero
        if std == 0 or np.isnan(std):
            df['z_' + c] = 0.0
        else:
            df['z_' + c] = (df[c] - mean) / std

    # Final set of columns returned (exact names used in modeling)
    final_cols = [
        'accept',
        'female',
        'black',
        'self_employed',
        'married',
        'bad_history',
        'denied_PMI',
        'z_housing_expense_ratio',
        'z_PI_ratio',
        'z_loan_to_value',
        'z_mortgage_credit',
        'z_consumer_credit'
    ]

    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a multivariable logistic regression to estimate the effect of gender (female)
    on loan acceptance (accept), adjusting for creditworthiness and other controls.

    Returns a dictionary containing:
    - 'model': the fitted statsmodels result object
    - 'odds_ratios': pandas Series of exponentiated coefficients (odds ratios)
    - 'conf_int_odds': DataFrame of 95% CI for odds ratios
    - 'summary_text': model.summary().as_text() for quick textual inspection
    - 'n_obs' : number of observations used
    - 'accept_rate_by_gender': a small DataFrame with acceptance rates by female
    
    The caller should pass the output of transform(df) into this function.
    """
    # Expect df already transformed (contains the exact columns listed in transform)
    df = df.copy()

    import statsmodels.api as sm
    import numpy as np
    import pandas as pd

    # Define predictors: focal variable female plus controls
    X_cols = [
        'female',
        'black',
        'self_employed',
        'married',
        'bad_history',
        'denied_PMI',
        'z_housing_expense_ratio',
        'z_PI_ratio',
        'z_loan_to_value',
        'z_mortgage_credit',
        'z_consumer_credit'
    ]

    # Safety check: ensure columns are present
    missing = [c for c in X_cols + ['accept'] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input dataframe required for modeling: {missing}")

    X = df[X_cols]
    X = sm.add_constant(X, has_constant='add')
    y = df['accept']

    # Fit logistic regression (binomial) using Maximum Likelihood
    logit_model = sm.Logit(y, X)
    try:
        res = logit_model.fit(disp=False)
    except Exception as e:
        # Retry with a small regularization if convergence problems arise
        res = logit_model.fit_regularized(method='l1', disp=False)

    # Compute odds ratios and 95% CI for odds ratios
    params = res.params
    conf = res.conf_int()
    conf.columns = ['2.5%', '97.5%']
    odds_ratios = np.exp(params)
    conf_odds = np.exp(conf)

    # Summary info for quick diagnostics
    summary_text = res.summary().as_text()
    n_obs = int(res.nobs)

    # Acceptance rates by gender (raw, unadjusted)
    accept_rate_by_gender = df.groupby('female')['accept'].agg(['mean', 'count']).rename(columns={'mean': 'accept_rate'})

    results = {
        'model': res,
        'odds_ratios': odds_ratios,
        'conf_int_odds': conf_odds,
        'summary_text': summary_text,
        'n_obs': n_obs,
        'accept_rate_by_gender': accept_rate_by_gender
    }

    return results


