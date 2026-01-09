from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/.venv/lib/python3.10/site-packages/blade_bench/datasets/mortgage/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe ready for modeling.

    Produces the following columns used by the model:
      - approved: 1 if mortgage was approved, 0 if denied (derived as 1 - mortgage_credit)
      - female: 1 if applicant is female, 0 if male (from consumer_credit)
      - black: 1 if applicant is Black, 0 otherwise (from bad_history)
      - married: 1 if married, 0 otherwise (from married)
      - self_employed: 1 if self-employed, 0 otherwise (from self_employed)
      - credit_score_z: standardized consumer/credit score (from accept)
      - loan_to_value_z: standardized loan_to_value
      - debt_to_income_z: standardized denied_PMI (used here as a debt/expense proxy)

    The function handles missing values for numeric controls by median imputation and coerces types.
    """
    df = df.copy()

    # Ensure required raw columns exist
    # Drop rows missing the key variables for gender and mortgage outcome
    if 'consumer_credit' not in df.columns or 'mortgage_credit' not in df.columns:
        # If the expected columns are missing, raise a clear error
        missing = [c for c in ['consumer_credit', 'mortgage_credit'] if c not in df.columns]
        raise KeyError(f"Missing required column(s) for transformation: {missing}")

    df = df.dropna(subset=['consumer_credit', 'mortgage_credit']).copy()

    # Create female indicator from consumer_credit (metadata: 1 if female, 0 if male)
    # Coerce to numeric then to 0/1 integers (rounding just in case values are floats near 0/1)
    df['female'] = pd.to_numeric(df['consumer_credit'], errors='coerce').round().fillna(0).astype(int)

    # Create approved outcome: metadata documents 'mortgage_credit' as 1=denied, 0=accepted
    df['mortgage_credit'] = pd.to_numeric(df['mortgage_credit'], errors='coerce')
    # If any values are outside {0,1} treat them carefully by thresholding at 0.5 after coercion
    df['mortgage_credit_bin'] = df['mortgage_credit'].apply(lambda x: 1 if x >= 0.5 else 0)
    df['approved'] = (1 - df['mortgage_credit_bin']).astype(int)
    df.drop(columns=['mortgage_credit_bin'], inplace=True)

    # Controls: create/clean columns if present, otherwise create defaults
    if 'bad_history' in df.columns:
        df['black'] = pd.to_numeric(df['bad_history'], errors='coerce').round().fillna(0).astype(int)
    else:
        df['black'] = 0

    if 'married' in df.columns:
        df['married'] = pd.to_numeric(df['married'], errors='coerce').round().fillna(0).astype(int)
    else:
        df['married'] = 0

    if 'self_employed' in df.columns:
        df['self_employed'] = pd.to_numeric(df['self_employed'], errors='coerce').round().fillna(0).astype(int)
    else:
        df['self_employed'] = 0

    # Numeric continuous controls: credit score, loan_to_value, debt proxy
    # Per metadata, 'accept' corresponds to applicant's consumer credit score
    if 'accept' in df.columns:
        df['credit_score'] = pd.to_numeric(df['accept'], errors='coerce')
    else:
        # fallback: try 'consumer_credit' (but this is binary); if not available set median later
        df['credit_score'] = pd.to_numeric(df.get('credit_score', pd.Series([np.nan]*len(df), index=df.index)), errors='coerce')

    if 'loan_to_value' in df.columns:
        df['loan_to_value'] = pd.to_numeric(df['loan_to_value'], errors='coerce')
    else:
        df['loan_to_value'] = pd.Series([np.nan]*len(df), index=df.index)

    # 'denied_PMI' is used here as a continuous expense/debt proxy per the provided schema
    if 'denied_PMI' in df.columns:
        df['debt_to_income'] = pd.to_numeric(df['denied_PMI'], errors='coerce')
    else:
        df['debt_to_income'] = pd.Series([np.nan]*len(df), index=df.index)

    # Median imputation for continuous controls
    for col in ['credit_score', 'loan_to_value', 'debt_to_income']:
        if df[col].isnull().any():
            median_val = df[col].median(skipna=True)
            # If median is nan (entire column missing), fill with 0
            if np.isnan(median_val):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)

    # Standardize continuous controls to z-scores for model stability
    for col in ['credit_score', 'loan_to_value', 'debt_to_income']:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            # If no variance, create zero z-score
            df[col + '_z'] = 0.0
        else:
            df[col + '_z'] = (df[col] - mean) / std

    # Keep only the final columns used in modeling and return
    final_cols = [
        'approved',
        'female',
        'black',
        'married',
        'self_employed',
        'credit_score_z',
        'loan_to_value_z',
        'debt_to_income_z'
    ]

    # Ensure all final columns exist (they should after the steps above)
    for c in final_cols:
        if c not in df.columns:
            df[c] = 0

    return df[final_cols].reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a logistic regression predicting mortgage approval from gender and controls.

    Returns a dictionary with:
      - 'model': the fitted statsmodels Logit results instance
      - 'odds_ratios': pandas Series of exponentiated coefficients (odds ratios)
      - 'conf_odds': pandas DataFrame with exponentiated 95% confidence intervals

    Model specification:
      approved ~ female + black + married + self_employed + credit_score_z + loan_to_value_z + debt_to_income_z

    The function expects the dataframe produced by transform(...) which contains the columns:
      ['approved','female','black','married','self_employed','credit_score_z','loan_to_value_z','debt_to_income_z']
    """
    import statsmodels.api as sm

    # Check required columns
    required = ['approved','female','black','married','self_employed','credit_score_z','loan_to_value_z','debt_to_income_z']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for modeling: {missing}")

    # Outcome and predictors
    y = df['approved'].astype(int)
    X = df[['female', 'black', 'married', 'self_employed', 'credit_score_z', 'loan_to_value_z', 'debt_to_income_z']].astype(float)
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression (use robust handling of perfect separation if it occurs)
    try:
        logit_res = sm.Logit(y, X).fit(disp=False)
    except Exception as e:
        # If Logit fails (e.g., perfect separation), try using statsmodels GLM with binomial family
        glm_binom = sm.GLM(y, X, family=sm.families.Binomial())
        logit_res = glm_binom.fit()

    # Odds ratios and 95% CI for odds
    params = logit_res.params
    conf = logit_res.conf_int()
    conf.columns = ['2.5%', '97.5%']

    odds_ratios = np.exp(params)
    conf_odds = np.exp(conf)

    results = {
        'model': logit_res,
        'odds_ratios': odds_ratios,
        'conf_odds': conf_odds
    }

    return results


