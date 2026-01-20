from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/mortgage/anonymize_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw HMDA-derived dataset into a dataframe ready for logistic regression.

    Produces the following final columns (used in the model):
      - Approved: binary outcome (1 accepted, 0 denied)
      - Female: binary indicator (1 female, 0 male)
      - Black, SelfEmployed, Married, BadCreditHistory, PMI_denied: binary controls
      - feature1_std, feature4_std, MortgageScore_std, ConsumerScore_std, DebtToIncome_std, LTV_std: standardized numeric controls

    The function handles the presence of both feature11 (denied) and feature14 (accepted) by preferring feature14 when present.
    It drops rows with missing values in any of the final model columns.
    """
    # work on a copy
    df = df.copy()

    # Create Approved outcome: prefer feature14 (1 accepted, 0 denied) if present; else derive from feature11
    if 'feature14' in df.columns:
        df['Approved'] = pd.to_numeric(df['feature14'], errors='coerce').astype('Float64')
    elif 'feature11' in df.columns:
        # feature11: 1 if denied, 0 if accepted -> Approved = 1 - feature11
        df['Approved'] = 1 - pd.to_numeric(df['feature11'], errors='coerce').astype('Float64')
    else:
        raise KeyError('Neither feature14 nor feature11 present to construct Approved outcome')

    # Binary independent variable: Female (feature2: 1 if female, 0 if male)
    df['Female'] = pd.to_numeric(df['feature2'], errors='coerce').astype('Float64')

    # Binary controls
    df['Black'] = pd.to_numeric(df['feature3'], errors='coerce').astype('Float64')
    df['SelfEmployed'] = pd.to_numeric(df['feature5'], errors='coerce').astype('Float64')
    df['Married'] = pd.to_numeric(df['feature6'], errors='coerce').astype('Float64')
    df['BadCreditHistory'] = pd.to_numeric(df['feature9'], errors='coerce').astype('Float64')
    df['PMI_denied'] = pd.to_numeric(df['feature13'], errors='coerce').astype('Float64')

    # Continuous controls: convert to numeric and then standardize (z-score). Use ddof=0 for population-like std.
    numeric_map = {
        'feature1': 'feature1_std',
        'feature4': 'feature4_std',
        'feature7': 'MortgageScore_std',
        'feature8': 'ConsumerScore_std',
        'feature10': 'DebtToIncome_std',
        'feature12': 'LTV_std'
    }

    for raw_col, std_col in numeric_map.items():
        if raw_col not in df.columns:
            # If a numeric control is missing, create column with NaN so it will be dropped later
            df[std_col] = pd.NA
            continue
        # coerce to numeric
        series = pd.to_numeric(df[raw_col], errors='coerce').astype('Float64')
        # compute mean and std excluding NaNs
        mean = series.mean()
        std = series.std(ddof=0)
        if pd.isna(std) or std == 0:
            # If constant or all NaN, result will be NaN; keep column to allow downstream inspection
            df[std_col] = (series - mean)
        else:
            df[std_col] = (series - mean) / std

    # Final columns to be used in the model
    final_cols = [
        'Approved', 'Female', 'Black', 'SelfEmployed', 'Married', 'BadCreditHistory', 'PMI_denied',
        'feature1_std', 'feature4_std', 'MortgageScore_std', 'ConsumerScore_std', 'DebtToIncome_std', 'LTV_std'
    ]

    # Drop rows with missing values in any final column (list of required predictors and outcome)
    df = df.dropna(subset=final_cols)

    # Cast binary columns to integers (0/1) where appropriate
    for bcol in ['Approved', 'Female', 'Black', 'SelfEmployed', 'Married', 'BadCreditHistory', 'PMI_denied']:
        df[bcol] = df[bcol].astype(int)

    # Ensure standardized columns are float
    for c in ['feature1_std', 'feature4_std', 'MortgageScore_std', 'ConsumerScore_std', 'DebtToIncome_std', 'LTV_std']:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype('float')

    # Return only the columns likely needed downstream (keeps original data intact in the calling environment if needed)
    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (binomial logit) to estimate the effect of gender on mortgage approval,
    controlling for observed applicant and loan characteristics.

    Model specification (logit):
      Approved ~ Female + Black + SelfEmployed + Married + BadCreditHistory + PMI_denied
                 + feature1_std + feature4_std + MortgageScore_std + ConsumerScore_std
                 + DebtToIncome_std + LTV_std

    Returns the fitted statsmodels Logit result object.
    """
    # work on a copy of input
    df = df.copy()

    # Define outcome and predictors (must match the column names created in transform)
    y = df['Approved']

    X_cols = [
        'Female', 'Black', 'SelfEmployed', 'Married', 'BadCreditHistory', 'PMI_denied',
        'feature1_std', 'feature4_std', 'MortgageScore_std', 'ConsumerScore_std', 'DebtToIncome_std', 'LTV_std'
    ]

    X = df[X_cols]

    # Add constant
    X = sm.add_constant(X, has_constant='skip')

    # Fit logistic regression (use Logit; if convergence problems occur consider sm.GLM with family=sm.families.Binomial())
    logit_model = sm.Logit(y, X)
    results = logit_model.fit(disp=False)

    # For convenience, attach odds ratios and 95% CI to the results object
    params = results.params
    conf = results.conf_int()
    odds_ratios = np.exp(params)
    conf_odds = np.exp(conf)
    results.odds_ratios = odds_ratios
    results.conf_odds = conf_odds

    return results


