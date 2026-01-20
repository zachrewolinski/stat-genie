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
    Transform the raw Boston mortgage dataset into an analysis-ready dataframe.

    - Rename input columns to meaningful names.
    - Drop rows with missing values in variables required for the analysis.
    - Ensure binary variables are integer 0/1.
    - Create standardized (z-scored) continuous controls to aid model convergence and interpretation.
    - Return dataframe that contains all columns listed in the conceptual variables.
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Rename columns from schema to analysis-friendly names
    rename_map = {
        'feature1': 'loan_amount',        # numeric (continuous)
        'feature2': 'female',             # 1 if female, 0 if male
        'feature3': 'black',              # 1 if black, 0 otherwise
        'feature4': 'housing_ratio',      # housing expense / income
        'feature5': 'self_employed',      # 1/0
        'feature6': 'married',            # 1/0
        'feature7': 'mortgage_score',     # ordinal/score
        'feature8': 'consumer_score',     # ordinal/score
        'feature9': 'bad_credit',         # 1 if history of bad credit
        'feature10': 'debt_to_income',    # continuous ratio
        'feature11': 'denied_flag',       # 1 if denied (redundant with feature14)
        'feature12': 'ltv',               # loan-to-value ratio
        'feature13': 'pmi_denied',        # 1 if PMI denied
        'feature14': 'accepted'           # 1 if accepted, 0 if denied
    }
    df = df.rename(columns=rename_map)

    # Required columns for the analysis
    required_cols = [
        'accepted', 'female', 'black', 'loan_amount', 'housing_ratio', 'self_employed',
        'married', 'mortgage_score', 'consumer_score', 'bad_credit',
        'debt_to_income', 'ltv', 'pmi_denied'
    ]

    # Drop rows with missing data in required columns
    df = df.dropna(subset=required_cols)

    # Make sure binary indicators are integer (0/1)
    binary_cols = ['accepted', 'female', 'black', 'self_employed', 'married', 'bad_credit', 'pmi_denied']
    for c in binary_cols:
        # Some datasets use floats like 0.0/1.0; cast safely to int
        df[c] = df[c].astype(float).round().astype(int)

    # For safety, ensure accepted is coded 0/1. If there is an alternative denial indicator (denied_flag),
    # prefer feature14 mapping (accepted). If accepted values are not 0/1, coerce to 0/1 via threshold.
    df['accepted'] = df['accepted'].apply(lambda x: 1 if float(x) >= 0.5 else 0)

    # Standardize continuous predictors (z-score). Use sample std (ddof=1) to match common statistical conventions.
    cont_to_z = {
        'loan_amount': 'loan_amt_z',
        'housing_ratio': 'housing_ratio_z',
        'debt_to_income': 'debt_to_income_z',
        'ltv': 'ltv_z'
    }
    for orig, zname in cont_to_z.items():
        col = df[orig].astype(float)
        mean = col.mean()
        std = col.std(ddof=1)
        if std == 0 or np.isnan(std):
            # If constant, set z to 0
            df[zname] = 0.0
        else:
            df[zname] = (col - mean) / std

    # Create final set of columns to keep (these are the columns referenced in the model)
    final_cols = [
        'accepted', 'female', 'black',
        'loan_amt_z', 'housing_ratio_z', 'self_employed', 'married',
        'mortgage_score', 'consumer_score', 'bad_credit', 'debt_to_income_z', 'ltv_z', 'pmi_denied'
    ]

    # If any expected final columns are missing (shouldn't be), raise an informative error
    missing = [c for c in final_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns after transform: {missing}")

    # Return only the final columns (preserves index)
    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (binomial GLM) predicting approval (accepted) from applicant gender
    controlling for credit and loan/applicant characteristics. Test whether race (black) moderates
    the gender effect by including the interaction female * black.

    Returns the fitted model result object (statsmodels GLMResultsWrapper or LogitResults).
    """
    import statsmodels.formula.api as smf

    # Ensure dataframe contains expected columns
    required = ['accepted', 'female', 'black', 'loan_amt_z', 'housing_ratio_z', 'self_employed',
                'married', 'mortgage_score', 'consumer_score', 'bad_credit', 'debt_to_income_z', 'ltv_z', 'pmi_denied']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for modeling: {missing}")

    # Specify the formula: include main controls and the interaction female:black
    formula = (
        'accepted ~ female * black '
        '+ loan_amt_z + housing_ratio_z + self_employed + married '
        '+ mortgage_score + consumer_score + bad_credit '
        '+ debt_to_income_z + ltv_z + pmi_denied'
    )

    # Fit logistic regression using GLM with binomial family for stable results and easy robust cov options
    model_result = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Return the fitted model object (has summary(), params, conf_int(), etc.)
    return model_result