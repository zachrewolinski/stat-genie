from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# If you want to run this file directly, you can uncomment and adjust the path:
# df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/noperturb_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Boston HMDA-style dataset into a modeling dataframe.

    Produces the following columns used in the model (final dataframe):
      - accept (DV): 0/1 indicator (1 = accepted)
      - female (IV): 0/1
      - black, self_employed, married, bad_history, denied_PMI: 0/1
      - housing_exp_ratio_z, PI_ratio_z, loan_to_value_z, mortgage_credit_z, consumer_credit_z: standardized continuous controls

    Steps:
      - Keep only rows with non-missing values in the columns we will use
      - Ensure binary columns are ints (0/1)
      - Standardize continuous predictors (z-score) and name them with the exact final column names
    """
    # Make a shallow copy to avoid modifying caller's df
    df = df.copy()

    # Raw input column names expected in the source data
    # Note: we map raw continuous column names to the required final standardized names
    required_raw_cols = [
        'accept', 'female', 'black', 'housing_expense_ratio', 'self_employed', 'married',
        'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio', 'loan_to_value', 'denied_PMI'
    ]

    missing = [c for c in required_raw_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    # Keep only rows that have non-missing values for all required modeling raw columns
    df = df.dropna(subset=required_raw_cols)

    # Ensure binary columns are integers 0/1
    bin_cols = ['accept', 'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI']
    for c in bin_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)
        if df[c].isnull().any():
            df = df.dropna(subset=[c])
        df[c] = df[c].astype(int)

    # Continuous predictors: raw names and their corresponding final z-scaled column names
    cont_raw = ['housing_expense_ratio', 'PI_ratio', 'loan_to_value', 'mortgage_credit', 'consumer_credit']
    cont_final_z = ['housing_exp_ratio_z', 'PI_ratio_z', 'loan_to_value_z', 'mortgage_credit_z', 'consumer_credit_z']

    # Convert raw continuous columns to numeric and drop rows with missing continuous values
    for c in cont_raw:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=cont_raw)

    # Standardize continuous controls (z-score). Name columns with the exact final names required.
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df[cont_raw])
    df_scaled = pd.DataFrame(scaled_array, columns=cont_final_z, index=df.index)

    # Attach scaled columns to df with the required final column names
    for col in df_scaled.columns:
        df[col] = df_scaled[col]

    # Final column subset that will be used in the model: ensure order and exact required names
    final_cols = [
        'accept', 'female', 'black', 'housing_exp_ratio_z', 'self_employed', 'married',
        'mortgage_credit_z', 'consumer_credit_z', 'bad_history', 'PI_ratio_z', 'loan_to_value_z', 'denied_PMI'
    ]

    # If any final columns are missing (should not be), raise
    missing2 = [c for c in final_cols if c not in df.columns]
    if missing2:
        raise ValueError(f"Unexpected missing columns after transform: {missing2}")

    # Return only the final columns (plus original index)
    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Run a logistic regression (logit) predicting accept (1 = accepted) from female and controls.

    Model specification (primary):
      accept ~ female + black + self_employed + married + bad_history + denied_PMI
               + housing_exp_ratio_z + PI_ratio_z + loan_to_value_z + mortgage_credit_z + consumer_credit_z

    Returns:
      - dict containing:
        - 'result': the fitted statsmodels result object (Logit) or GLM if Logit fails to converge / errors
        - 'odds_ratios_table': DataFrame with odds ratios, 95% CI, and p-values
    """
    import numpy as _np

    # Ensure required columns exist in the provided df
    required = [
        'accept', 'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI',
        'housing_exp_ratio_z', 'PI_ratio_z', 'loan_to_value_z', 'mortgage_credit_z', 'consumer_credit_z'
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Model dataframe missing columns: {missing}")

    # Dependent and independent variables
    y = df['accept'].astype(int)

    X_cols = [
        'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI',
        'housing_exp_ratio_z', 'PI_ratio_z', 'loan_to_value_z', 'mortgage_credit_z', 'consumer_credit_z'
    ]
    X = df[X_cols].astype(float)

    # Add constant for intercept
    X = sm.add_constant(X, has_constant='add')

    # Try Logit first; if it fails (perfect separation or convergence), fall back to GLM with binomial
    try:
        logit_mod = sm.Logit(y, X)
        result = logit_mod.fit(disp=False, maxiter=100)
    except Exception:
        glm_mod = sm.GLM(y, X, family=sm.families.Binomial())
        result = glm_mod.fit()

    # Print concise summary
    print(result.summary())

    # Compute odds ratios and 95% CI
    params = result.params
    conf = result.conf_int()
    odds_ratios = _np.exp(params)
    conf_odds = _np.exp(conf)

    or_table = (pd.DataFrame({
        'OR': odds_ratios,
        'CI_lower': conf_odds[0],
        'CI_upper': conf_odds[1],
        'pvalue': result.pvalues
    }))

    print('\nOdds ratios with 95% CI:')
    print(or_table)

    # Return the fitted result and the OR table for further programmatic use
    return {
        'result': result,
        'odds_ratios_table': or_table
    }