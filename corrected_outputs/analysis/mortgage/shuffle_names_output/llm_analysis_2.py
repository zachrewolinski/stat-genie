from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/mortgage/shuffle_names_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy to avoid side effects
    df = df.copy()

    # Create outcome: Approved = 1 if application accepted, 0 if denied.
    # According to the provided schema, 'mortgage_credit' is coded 1 = denied, 0 = accepted.
    # If that assumption is wrong for your file, inspect values and invert accordingly.
    df['mortgage_credit'] = pd.to_numeric(df.get('mortgage_credit'), errors='coerce')
    df['Approved'] = df['mortgage_credit'].apply(lambda x: 1 if x == 0 else (0 if x == 1 else np.nan))

    # Create Female indicator from the provided 'consumer_credit' column which schema says encodes sex
    # (1 if applicant is female, 0 if male). Coerce to numeric and force to 0/1 where possible.
    df['Female'] = pd.to_numeric(df.get('consumer_credit'), errors='coerce')

    # Controls: coerce to numeric and create columns with clear names.
    df['CreditScore'] = pd.to_numeric(df.get('accept'), errors='coerce')
    df['LoanToValue'] = pd.to_numeric(df.get('loan_to_value'), errors='coerce')
    df['DebtToIncome'] = pd.to_numeric(df.get('denied_PMI'), errors='coerce')
    df['HousingExpenseRatio'] = pd.to_numeric(df.get('housing_expense_ratio'), errors='coerce')
    df['SelfEmployed'] = pd.to_numeric(df.get('self_employed'), errors='coerce')
    df['Married'] = pd.to_numeric(df.get('married'), errors='coerce')
    # According to schema 'bad_history' is described as 1 if applicant is Black, 0 otherwise.
    df['Black'] = pd.to_numeric(df.get('bad_history'), errors='coerce')

    # Keep only rows with non-missing values for the variables used in the model
    required_cols = ['Approved', 'Female', 'CreditScore', 'LoanToValue', 'DebtToIncome',
                     'HousingExpenseRatio', 'SelfEmployed', 'Married', 'Black']
    df = df.dropna(subset=required_cols)

    # Ensure binary columns are integer 0/1
    for bin_col in ['Female', 'LoanToValue', 'SelfEmployed', 'Married', 'Black']:
        # round or force to 0/1 where appropriate
        df[bin_col] = df[bin_col].round().astype(int)

    # Optionally: keep only relevant columns for the analysis to make downstream use explicit
    df = df[required_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # Build the design matrix for logistic regression predicting approval
    df = df.copy()

    # Independent variable and controls
    X_cols = ['Female', 'CreditScore', 'LoanToValue', 'DebtToIncome',
              'HousingExpenseRatio', 'SelfEmployed', 'Married', 'Black']

    X = df[X_cols]
    # Add constant for intercept
    X = sm.add_constant(X, has_constant='add')

    y = df['Approved']

    # Fit logistic regression (maximum likelihood)
    logit_model = sm.Logit(y, X)
    # suppress optimizer output by disp=False
    results = logit_model.fit(disp=False)

    # For easier interpretation, also attach odds ratios and 95% CIs to results.summary-like output
    params = results.params
    conf = results.conf_int()
    odds_ratios = np.exp(params)
    conf_odds = np.exp(conf)

    # Attach these to results for downstream inspection
    results.odds_ratios = odds_ratios
    results.conf_odds = conf_odds

    return results


