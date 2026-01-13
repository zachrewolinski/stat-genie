from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/shuffle_names_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a cleaned dataframe containing the variables used in the model.

    The function creates:
      - female: binary indicator (1 if female, 0 if male) derived from 'consumer_credit' if present (per dataset description),
                otherwise attempts to use an existing 'female' column.
      - approved: binary indicator (1 if application approved, 0 if denied) derived from 'mortgage_credit' (1=denied -> invert)
                  or from 'Unnamed: 0' if it encodes acceptance directly.
      - credit_score: mapped from 'accept' column if present (dataset's labeling is inconsistent; 'accept' appears to be credit score code).
      - and ensures controls: loan_to_value, housing_expense_ratio, denied_PMI, married, self_employed, bad_history

    The function coerces types to numeric, handles common encoding issues, drops rows with missing values in the final model columns,
    and returns a dataframe that contains exactly the columns used in the statistical model.
    """
    df = df.copy()

    # Create female indicator
    if 'consumer_credit' in df.columns:
        # dataset schema indicates 'consumer_credit' == 1 if female, 0 if male
        df['female'] = pd.to_numeric(df['consumer_credit'], errors='coerce').round().fillna(0).astype(int)
    elif 'female' in df.columns:
        # try to coerce to 0/1
        col = pd.to_numeric(df['female'], errors='coerce')
        # Map values > 0.5 to 1, else 0
        df['female'] = (col > 0.5).astype(int).fillna(0)
    else:
        # If no gender column present, create NA column so missing rows will be dropped later
        df['female'] = np.nan

    # Create approved outcome indicator
    # Prefer 'mortgage_credit' (documentation: 1 = denied, 0 = accepted) -> approved = 1 - mortgage_credit
    if 'mortgage_credit' in df.columns:
        mc = pd.to_numeric(df['mortgage_credit'], errors='coerce')
        # If mortgage_credit encodes denial as 1, approved = 1 - mortgage_credit
        df['approved'] = (1 - mc.round()).clip(0, 1)
    elif 'Unnamed: 0' in df.columns:
        # documentation suggested Unnamed: 0 might be 1 = accepted, 0 = denied
        u0 = pd.to_numeric(df['Unnamed: 0'], errors='coerce')
        df['approved'] = u0.round().clip(0, 1)
    else:
        # Fallback: try to infer from columns with 'deny' or similar; if not possible, create NA
        df['approved'] = np.nan
        if 'deny' in df.columns:
            # If 'deny' counts or codes nonzero -> treat >0 as denied
            deny = pd.to_numeric(df['deny'], errors='coerce')
            # If deny is binary and 1 means denied, approved = 1 - deny
            if deny.dropna().isin([0, 1]).all():
                df['approved'] = (1 - deny).clip(0, 1)

    # Controls: credit_score (from 'accept' in provided schema), loan_to_value, housing_expense_ratio, denied_PMI, married, self_employed, bad_history
    # Coerce to numeric where appropriate
    if 'accept' in df.columns:
        df['credit_score'] = pd.to_numeric(df['accept'], errors='coerce')
    else:
        df['credit_score'] = np.nan

    for col in ['loan_to_value', 'housing_expense_ratio', 'denied_PMI']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan

    for col in ['married', 'self_employed', 'bad_history']:
        if col in df.columns:
            # map to 0/1 if possible
            temp = pd.to_numeric(df[col], errors='coerce')
            # If not 0/1, treat >0.5 as 1
            if not temp.dropna().isin([0, 1]).all():
                df[col] = (temp > 0.5).astype(float)
            else:
                df[col] = temp.astype(float)
        else:
            df[col] = np.nan

    # Select the final columns used in modeling
    final_cols = [
        'female',
        'approved',
        'credit_score',
        'loan_to_value',
        'housing_expense_ratio',
        'denied_PMI',
        'married',
        'self_employed',
        'bad_history'
    ]

    # Drop rows with missing values in any of the final columns -- necessary for regression
    df_final = df[final_cols].copy()

    # If credit_score appears categorical (1-6), keep as is. For continuous controls, keep numeric values.
    # Drop rows where approved or female are missing since these are essential
    df_final = df_final.dropna(subset=['female', 'approved'])

    # For model stability, also drop rows with missing values in the main controls
    df_final = df_final.dropna(subset=['credit_score', 'loan_to_value', 'housing_expense_ratio', 'denied_PMI', 'married', 'self_employed', 'bad_history'])

    # Coerce types to appropriate dtypes
    df_final['female'] = df_final['female'].astype(int)
    df_final['approved'] = df_final['approved'].astype(int)
    df_final['married'] = df_final['married'].astype(int)
    df_final['self_employed'] = df_final['self_employed'].astype(int)
    df_final['bad_history'] = df_final['bad_history'].astype(int)

    # Ensure numeric columns are floats
    for col in ['credit_score', 'loan_to_value', 'housing_expense_ratio', 'denied_PMI']:
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce').astype(float)

    # Return the dataframe that contains exactly the columns referenced by the model
    return df_final


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression predicting mortgage approval from applicant gender and controls.

    Model specification:
      approved ~ female + credit_score + loan_to_value + housing_expense_ratio + denied_PMI + married + self_employed + bad_history

    Returns the fitted statsmodels results object (Logit result). Also prints a concise summary and the odds ratios with 95% CIs.
    """
    # Ensure we have the columns the transform function should have produced
    required = ['female', 'approved', 'credit_score', 'loan_to_value', 'housing_expense_ratio', 'denied_PMI', 'married', 'self_employed', 'bad_history']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Prepare design matrix
    X = df[['female', 'credit_score', 'loan_to_value', 'housing_expense_ratio', 'denied_PMI', 'married', 'self_employed', 'bad_history']].copy()
    X = sm.add_constant(X, has_constant='add')
    y = df['approved']

    # Fit logistic regression (maximum likelihood)
    # Use try/except to catch potential convergence issues
    try:
        logit_model = sm.Logit(y, X)
        results = logit_model.fit(disp=False)
    except Exception as e:
        # Attempt a penalized fit (L1/L2) or report error
        # Here we try a simple .fit_regularized if available to gain convergence
        try:
            results = logit_model.fit_regularized(method='l1')
        except Exception:
            raise RuntimeError(f"Logistic regression failed to converge: {e}")

    # Print summary (user-facing); computing odds ratios for interpretability
    print(results.summary())

    # Compute odds ratios and 95% CIs
    params = results.params
    conf = results.conf_int()
    odds_ratios = np.exp(params)
    conf_odds = np.exp(conf)
    or_table = pd.DataFrame({
        'OR': odds_ratios,
        'CI_lower': conf_odds[0],
        'CI_upper': conf_odds[1]
    })
    print('\nOdds ratios with 95% CIs:\n', or_table)

    # Return the fitted results object for further inspection
    return results


