from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/replace_with_rvs_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for logistic regression of mortgage acceptance on applicant gender.

    Transformations performed:
    - Make a copy of the input dataframe.
    - Ensure necessary columns exist and coerce them to appropriate numeric types.
    - Drop rows with missing values in dependent variable, independent variable, or controls.
    - Ensure binary indicators are integer typed.

    The returned dataframe contains the same column names used by the model:
      'accept', 'female', 'black', 'mortgage_credit', 'consumer_credit',
      'bad_history', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio',
      'self_employed', 'married', 'denied_PMI'
    """
    df = df.copy()

    # Required columns for the analysis
    required_cols = [
        'accept', 'female', 'black', 'mortgage_credit', 'consumer_credit',
        'bad_history', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio',
        'self_employed', 'married', 'denied_PMI'
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    # Coerce numeric columns (will convert strings that look numeric)
    # Binary indicators -- allow coercion; later we'll drop rows with NaNs
    binary_cols = ['female', 'black', 'bad_history', 'self_employed', 'married', 'denied_PMI', 'accept']
    for c in binary_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Continuous / ordinal numeric columns
    numeric_cols = ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows with missing values in any variable used by the model
    model_cols = required_cols.copy()
    df = df.dropna(subset=model_cols)

    # Ensure binary columns are integer (0/1)
    df[binary_cols] = df[binary_cols].astype(int)

    # Keep the final set of columns (no renaming) - model expects these exact names
    final_cols = required_cols
    df = df[final_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression predicting mortgage acceptance ('accept') using applicant gender ('female')
    and the listed controls. Returns the fitted statsmodels result object.

    Model specification (logit):
      accept ~ female + black + mortgage_credit + consumer_credit + bad_history
               + PI_ratio + loan_to_value + housing_expense_ratio + self_employed
               + married + denied_PMI

    The function also constructs and returns a small dictionary with odds ratios and 95% CIs
    for convenience.
    """
    import statsmodels.api as sm

    # Ensure required columns are present
    cols = [
        'female', 'black', 'mortgage_credit', 'consumer_credit', 'bad_history',
        'PI_ratio', 'loan_to_value', 'housing_expense_ratio', 'self_employed',
        'married', 'denied_PMI'
    ]
    for c in cols + ['accept']:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found in dataframe")

    X = df[cols]
    X = sm.add_constant(X, has_constant='add')
    y = df['accept']

    # Fit logistic regression (maximum likelihood)
    try:
        logit_model = sm.Logit(y, X)
        result = logit_model.fit(disp=False, maxiter=200)
    except Exception:
        # fallback to GLM with binomial family if Logit has convergence issues
        glm_binom = sm.GLM(y, X, family=sm.families.Binomial())
        result = glm_binom.fit()

    # Compute odds ratios and 95% CI
    params = result.params
    conf = result.conf_int()
    odds_ratios = np.exp(params)
    conf_odds = np.exp(conf)

    summary_dict = {
        'model_result': result,
        'odds_ratios': odds_ratios.to_dict() if hasattr(odds_ratios, 'to_dict') else odds_ratios,
        'odds_CI_lower': conf_odds[0].to_dict() if hasattr(conf_odds, 'to_dict') else conf_odds[0],
        'odds_CI_upper': conf_odds[1].to_dict() if hasattr(conf_odds, 'to_dict') else conf_odds[1]
    }

    return summary_dict


