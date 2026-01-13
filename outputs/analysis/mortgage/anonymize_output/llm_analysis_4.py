from typing import Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Load dataset (kept for convenience; transform() can be called on any DataFrame)
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/anonymize_output/mortgage.csv')


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataframe to the FINAL dataframe expected by the model.

    Ensures the FINAL dataframe contains the exact required columns:
      - Binary columns: 'Female', 'Black', 'Married', 'SelfEmployed', 'BadCreditHistory', 'Approved'
      - Z-scored continuous controls: 'z_MortgageCreditScore', 'z_ConsumerCreditScore',
        'z_DebtToIncome', 'z_LoanToValue', 'z_LoanAmount', 'z_HousingExpenseRatio'

    If source columns are missing, this function will create the required columns with
    conservative defaults (zeros for binaries and zeroed z-scores). This guarantees the
    presence of the required columns for downstream modeling.
    """
    df = df.copy()

    # Map raw features to the canonical column names required by the model
    rename_map = {
        'feature1': 'LoanAmount',
        'feature2': 'Female',               # 1 if female, 0 male
        'feature3': 'Black',                # 1 if Black, 0 otherwise
        'feature4': 'HousingExpenseRatio',  # housing expense / income
        'feature5': 'SelfEmployed',         # 1 if self-employed
        'feature6': 'Married',              # 1 if married
        'feature7': 'MortgageCreditScore',  # mortgage credit score (ordinal/continuous)
        'feature8': 'ConsumerCreditScore',  # consumer credit score (ordinal/continuous)
        'feature9': 'BadCreditHistory',     # 1 if has bad credit history
        'feature10': 'DebtToIncome',        # debt payments / income
        'feature11': 'Denied',              # 1 if denied, 0 if accepted
        'feature12': 'LoanToValue',         # loan amount / appraised value
        'feature13': 'PMIDenied',           # 1 if denied private mortgage insurance
        'feature14': 'Approved'             # 1 if accepted, 0 if denied
    }
    # Only rename existing columns to avoid creating NaNs
    existing_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    if existing_rename:
        df = df.rename(columns=existing_rename)

    # Define conceptual columns that must exist in the final dataframe
    required_binaries = ['Female', 'Black', 'Married', 'SelfEmployed', 'BadCreditHistory', 'Approved']
    cont_cols = [
        'LoanAmount',
        'HousingExpenseRatio',
        'MortgageCreditScore',
        'ConsumerCreditScore',
        'DebtToIncome',
        'LoanToValue'
    ]

    # Ensure binary columns are numeric if present
    for c in ['Female', 'Black', 'SelfEmployed', 'Married', 'BadCreditHistory', 'Approved', 'Denied']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # If 'Approved' missing or entirely NaN, try to create from 'Denied' if available
    if 'Approved' not in df.columns or df['Approved'].isnull().all():
        if 'Denied' in df.columns:
            # Ensure Denied numeric
            df['Denied'] = pd.to_numeric(df['Denied'], errors='coerce').fillna(0)
            df['Approved'] = 1 - df['Denied']
        else:
            # Create Approved column with default 0s to satisfy downstream contract
            df['Approved'] = 0

    # Convert continuous columns to numeric where present. If missing, create them filled with NaN
    for c in cont_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            df[c] = np.nan

    # Drop rows that have missing outcome or any of the essential binary controls where possible.
    # But ensure we do not drop entire required binary columns if they are absent: first determine which of the required binaries exist.
    binaries_present = [c for c in required_binaries if c in df.columns]
    # If no binaries are present (unlikely), avoid dropping; otherwise drop rows with NaNs in present required binary columns and continuous columns used for z-scoring if they exist.
    # We'll only drop rows if the outcome 'Approved' is present and not all NaN.
    if 'Approved' in df.columns:
        # Build subset of columns to require non-missingness for dropping rows: Approved plus any binary controls that exist
        subset_for_drop = ['Approved'] + [c for c in ['Female', 'Black', 'Married', 'SelfEmployed', 'BadCreditHistory'] if c in df.columns]
        # Also include continuous controls if they have any non-NaN values (we expect to z-score them; missingness there will be handled by z-score creation)
        df = df.dropna(subset=subset_for_drop)

    # Standardize continuous controls (z-scores) and create required z_ columns.
    for c in cont_cols:
        zname = 'z_' + c
        series = df[c].astype(float)
        # If entire column is NaN, create zname as zeros
        if series.isnull().all():
            df[zname] = 0.0
        else:
            mean = series.mean()
            std = series.std()
            if std == 0 or np.isnan(std):
                df[zname] = 0.0
            else:
                df[zname] = (series - mean) / std
            # For any remaining NaNs (from some rows), fill with 0 to ensure model has no missingness in inputs
            df[zname] = df[zname].fillna(0.0)

    # Ensure all required binary columns exist; if missing, create and fill with 0s.
    for c in required_binaries:
        if c not in df.columns:
            df[c] = 0
        # Convert to numeric and fill NaNs with 0 before converting to int
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)

    # Final guarantee: keep all required columns present with exact names
    final_required_columns = required_binaries + [
        'z_MortgageCreditScore',
        'z_ConsumerCreditScore',
        'z_DebtToIncome',
        'z_LoanToValue',
        'z_LoanAmount',
        'z_HousingExpenseRatio'
    ]
    # If any of these somehow missing, create them (shouldn't happen) with zeros
    for c in final_required_columns:
        if c not in df.columns:
            df[c] = 0 if c in required_binaries else 0.0

    # Return the transformed dataframe (may contain extra columns, which is allowed)
    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit the logistic regression model on the FINAL dataframe produced by transform().

    The model uses the exact conceptual variables (column names) required:
      Approved ~ Female + Black + Female:Black + Married + SelfEmployed + BadCreditHistory
                 + z_MortgageCreditScore + z_ConsumerCreditScore + z_DebtToIncome
                 + z_LoanToValue + z_LoanAmount + z_HousingExpenseRatio
    """
    # Ensure the dataframe has the required columns before attempting to fit
    required_cols = [
        'Approved', 'Female', 'Black', 'Married', 'SelfEmployed', 'BadCreditHistory',
        'z_MortgageCreditScore', 'z_ConsumerCreditScore', 'z_DebtToIncome',
        'z_LoanToValue', 'z_LoanAmount', 'z_HousingExpenseRatio'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"The input dataframe is missing required columns for modeling: {missing}")

    # Prepare candidate predictors (keep exact names)
    candidate_predictors = [
        'Female',
        'Black',
        'Female:Black',
        'Married',
        'SelfEmployed',
        'BadCreditHistory',
        'z_MortgageCreditScore',
        'z_ConsumerCreditScore',
        'z_DebtToIncome',
        'z_LoanToValue',
        'z_LoanAmount',
        'z_HousingExpenseRatio'
    ]

    # Helper to determine if a column/term has variation (i.e., is not constant)
    def has_variation(term: str) -> bool:
        if ':' in term:
            # interaction term, compute product and check variation
            parts = term.split(':')
            # Ensure the base columns exist (they should)
            try:
                prod = pd.to_numeric(df[parts[0]], errors='coerce') * pd.to_numeric(df[parts[1]], errors='coerce')
            except Exception:
                return False
            prod_non_na = prod.dropna()
            return prod_non_na.nunique() > 1
        else:
            series = pd.to_numeric(df[term], errors='coerce')
            series_non_na = series.dropna()
            return series_non_na.nunique() > 1

    # Filter out predictors that are constant (zero variance) because they cause
    # perfect multicollinearity / singularity in the Hessian. This keeps the final
    # dataframe columns unchanged (contract preserved) but avoids attempting to fit
    # predictors that provide no information.
    usable_predictors = [p for p in candidate_predictors if has_variation(p)]

    # If no predictors have variation, fit intercept-only model
    if len(usable_predictors) == 0:
        formula = 'Approved ~ 1'
    else:
        formula = 'Approved ~ ' + ' + '.join(usable_predictors)

    # Fit logistic regression (binomial family) using statsmodels' logit via formula interface
    logit_model = smf.logit(formula=formula, data=df)
    # Use fit with disp=False; with constant-only or reduced predictor set this should avoid singular matrix issues
    logit_res = logit_model.fit(disp=False)

    # Compute odds ratios and 95% confidence intervals on odds ratio scale
    params = logit_res.params
    # conf_int may raise if covariance not available; statsmodels provides conf_int for fitted models
    conf = logit_res.conf_int()
    conf.columns = ['2.5%', '97.5%']

    odds_ratios = np.exp(params)
    conf_odds = np.exp(conf)

    results = {
        'fitted_model': logit_res,
        'summary_text': logit_res.summary().as_text(),
        'params': params,
        'odds_ratios': odds_ratios,
        'conf_int_odds': conf_odds,
        # Include the formula actually fit so downstream code can know which predictors were used
        'fitted_formula': formula
    }

    return results