from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/add_features_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataset for modeling the effect of gender on mortgage approval.
    - Coerce relevant columns to numeric
    - Drop rows with missing values in the dependent, independent, or control variables
    - Ensure binary indicators are integer-typed
    Returns the cleaned dataframe with the exact column names used in the model.
    """
    df = df.copy()

    # Columns required for the analysis (IV, DV, and controls)
    required_cols = [
        'accept', 'female', 'black', 'mortgage_credit', 'consumer_credit', 'bad_history',
        'PI_ratio', 'loan_to_value', 'married', 'self_employed', 'housing_expense_ratio',
        'denied_PMI', 'religiousness', 'occupation'
    ]

    # Coerce to numeric where possible (this will set non-convertible to NaN)
    for c in required_cols:
        # preserve original values for non-numeric columns -- convert coercively
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows with missing values in any required column
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    # Ensure binary indicator columns are integer 0/1
    for b in ['accept', 'female', 'black', 'bad_history', 'married', 'self_employed', 'denied_PMI']:
        # round then cast to int to guard against floats like 0.0/1.0
        df[b] = df[b].round().astype(int)

    # Keep only the columns we will use in the model (cleaned)
    # This also ensures the final dataframe contains exactly the column names listed in cvars
    final_cols = required_cols
    df = df[final_cols].copy()

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (binomial) model estimating the effect of applicant gender (female)
    on mortgage acceptance, controlling for observed applicant and loan characteristics.

    Model form (logit):
        logit(P(accept=1)) = alpha + beta_female*female + sum(gamma_k * control_k)

    Returns the fitted statsmodels LogitResults object. Also computes and prints odds ratios
    and 95% confidence intervals for interpretation.
    """
    import statsmodels.api as sm
    import numpy as np

    df = df.copy()

    # Columns used in the model (must match transform output)
    model_cols = [
        'female', 'black', 'mortgage_credit', 'consumer_credit', 'bad_history',
        'PI_ratio', 'loan_to_value', 'married', 'self_employed', 'housing_expense_ratio',
        'denied_PMI', 'religiousness', 'occupation'
    ]

    # Ensure no missingness (transform should have dropped NA already, but be safe)
    df = df.dropna(subset=['accept'] + model_cols)

    X = df[model_cols]
    X = sm.add_constant(X)
    y = df['accept']

    # Fit logistic regression
    logit = sm.Logit(y, X)
    results = logit.fit(disp=False)

    # Calculate odds ratios and 95% CI for easier interpretation
    params = results.params
    conf = results.conf_int()
    odds_ratios = np.exp(params)
    conf_odds = np.exp(conf)

    # Attach readable summary information to the results object (non-invasive)
    try:
        results.odds_ratios = odds_ratios
        results.conf_odds = conf_odds
    except Exception:
        # If the results object is immutable in some environment, return a dict instead
        return {
            'fit': results,
            'odds_ratios': odds_ratios,
            'conf_odds': conf_odds
        }

    return results


