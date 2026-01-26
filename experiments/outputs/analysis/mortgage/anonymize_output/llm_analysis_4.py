from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset into an analysis-ready dataframe.

    - Rename original feature columns to meaningful names.
    - Ensure binary columns are numeric (0/1).
    - Drop rows missing essential predictors or outcome.
    - Standardize continuous numeric controls (z-scores) so coefficients are comparable.

    Returns dataframe containing the columns used in the model (see conceptual variables):
      ['Approved', 'Female', 'ApplicantIncome_z', 'Black', 'HousingExpenseRatio_z',
       'SelfEmployed', 'Married', 'MortgageCreditScore_z', 'ConsumerCreditScore_z',
       'BadCreditHistory', 'DebtToIncomeRatio_z', 'LoanToValue_z', 'PMIDenied']
    """
    df = df.copy()

    # Rename features to clear names
    rename_map = {
        'feature1': 'ApplicantIncome',
        'feature2': 'Female',            # 1 if applicant is female, 0 if male
        'feature3': 'Black',             # 1 if applicant is Black, 0 otherwise
        'feature4': 'HousingExpenseRatio',
        'feature5': 'SelfEmployed',
        'feature6': 'Married',
        'feature7': 'MortgageCreditScore',
        'feature8': 'ConsumerCreditScore',
        'feature9': 'BadCreditHistory',
        'feature10': 'DebtToIncomeRatio',
        'feature11': 'Denied_flag',      # redundant with feature14 (kept for reference but not used)
        'feature12': 'LoanToValue',
        'feature13': 'PMIDenied',
        'feature14': 'Approved'          # 1 if accepted, 0 if denied
    }
    df = df.rename(columns=rename_map)

    # Convert to numeric where appropriate; coerce errors to NaN
    to_numeric_cols = ['ApplicantIncome', 'Female', 'Black', 'HousingExpenseRatio', 'SelfEmployed',
                       'Married', 'MortgageCreditScore', 'ConsumerCreditScore', 'BadCreditHistory',
                       'DebtToIncomeRatio', 'LoanToValue', 'PMIDenied', 'Approved']
    for c in to_numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing the outcome or the focal independent variable or essential continuous controls
    required_for_model = ['Approved', 'Female', 'MortgageCreditScore', 'ConsumerCreditScore',
                          'DebtToIncomeRatio', 'LoanToValue', 'ApplicantIncome']
    missing_required = [c for c in required_for_model if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns in input dataframe: {missing_required}")

    df = df.dropna(subset=required_for_model)

    # Ensure binary columns are 0/1 integers (where they exist). Handle NaN/inf safely.
    binary_cols = ['Female', 'Black', 'SelfEmployed', 'Married', 'BadCreditHistory', 'PMIDenied', 'Approved']
    for c in binary_cols:
        if c in df.columns:
            # Replace infinite values with NaN, round, fill NaN with 0, clip to [0,1], then convert to int
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)
            df[c] = df[c].round().fillna(0).clip(0, 1).astype(int)

    # Standardize continuous numeric covariates (z-scores). Use population std (ddof=0) for stability.
    cont_cols = ['MortgageCreditScore', 'ConsumerCreditScore', 'DebtToIncomeRatio', 'HousingExpenseRatio', 'LoanToValue', 'ApplicantIncome']
    for c in cont_cols:
        zname = c + '_z'
        if c in df.columns:
            col = pd.to_numeric(df[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
            mean = col.mean()
            std = col.std(ddof=0)
            if std == 0 or np.isnan(std):
                df[zname] = 0.0
            else:
                df[zname] = ((col - mean) / std).fillna(0.0)
        else:
            # If original continuous column missing, create a zeroed z-column to keep final schema consistent
            df[zname] = 0.0

    # Ensure all final required columns exist. If some optional binary controls were missing from input,
    # create them with default 0 values so the final dataframe always contains the required schema.
    keep_cols = ['Approved', 'Female', 'ApplicantIncome_z', 'Black', 'HousingExpenseRatio_z',
                 'SelfEmployed', 'Married', 'MortgageCreditScore_z', 'ConsumerCreditScore_z',
                 'BadCreditHistory', 'DebtToIncomeRatio_z', 'LoanToValue_z', 'PMIDenied']

    for col in keep_cols:
        if col not in df.columns:
            # For *_z continuous columns, create float zeros. For binaries, create integer zeros.
            if col.endswith('_z'):
                df[col] = 0.0
            else:
                # For safety, create binary/int columns as 0
                df[col] = 0
                # ensure integer dtype
                try:
                    df[col] = df[col].astype(int)
                except Exception:
                    df[col] = df[col].fillna(0).astype(int)

    # Finally, select and order the columns exactly as required
    df_final = df[keep_cols].reset_index(drop=True)

    return df_final


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression predicting mortgage approval from gender (Female) controlling
    for applicant and credit characteristics. Returns the fitted model object and
    interpretable summaries:
      - odds_ratio_female: estimated odds ratio for being female (exp(beta_female))
      - odds_ratio_ci_female: 95% CI for the odds ratio
      - ame_female: average marginal effect (difference in predicted probability when Female=1 vs Female=0, averaged across sample)

    Notes:
    - The function expects the transformed dataframe produced by transform(...) which contains
      standardized continuous covariates with *_z suffixes and binary indicators as described
      in the conceptual variables.
    """
    df = df.copy()

    # Check required columns
    required = ['Approved', 'Female', 'MortgageCreditScore_z', 'ConsumerCreditScore_z',
                'DebtToIncomeRatio_z', 'HousingExpenseRatio_z', 'LoanToValue_z', 'ApplicantIncome_z',
                'Black', 'SelfEmployed', 'Married', 'BadCreditHistory', 'PMIDenied']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Transformed dataframe is missing required columns for modeling: {missing}")

    # Ensure correct dtypes
    df['Approved'] = pd.to_numeric(df['Approved'], errors='coerce').fillna(0).astype(int)
    df['Female'] = pd.to_numeric(df['Female'], errors='coerce').fillna(0).astype(int)

    # Define design matrix
    controls = ['MortgageCreditScore_z', 'ConsumerCreditScore_z', 'DebtToIncomeRatio_z',
                'HousingExpenseRatio_z', 'LoanToValue_z', 'ApplicantIncome_z',
                'Black', 'SelfEmployed', 'Married', 'BadCreditHistory', 'PMIDenied']
    X = df[['Female'] + controls].astype(float)
    X = sm.add_constant(X, has_constant='add')
    y = df['Approved'].astype(int)

    # Fit logistic regression (maximum likelihood)
    logit = sm.Logit(y, X)
    result = logit.fit(disp=False)

    # Odds ratio and CI for Female
    params = result.params
    conf = result.conf_int()
    odds_ratio_female = float(np.exp(params['Female']))
    odds_ci_female = (float(np.exp(conf.loc['Female', 0])), float(np.exp(conf.loc['Female', 1])))

    # Compute average marginal effect (AME) of Female by comparing predicted probabilities
    X1 = X.copy()
    X0 = X.copy()
    X1['Female'] = 1.0
    X0['Female'] = 0.0
    p1 = result.predict(X1)
    p0 = result.predict(X0)
    ame_female = float((p1 - p0).mean())

    # Package results (include the fitted model object for further inspection)
    results = {
        'model_result': result,
        'odds_ratio_female': odds_ratio_female,
        'odds_ratio_ci_female': odds_ci_female,
        'ame_female': ame_female,
        'n_obs': int(result.nobs)
    }
    return results