from typing import Any, Dict, List
from types import SimpleNamespace
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from scipy import stats

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Boston mortgage dataset into the final dataframe used for modeling.

    Final columns created and returned:
      - Approved: 1 if application approved, 0 if denied
      - Female: 1 if female applicant, 0 if male
      - Black, BadHistory, LoanToValue, DeniedPMI, HousingExpenseRatio, SelfEmployed, Married, PI_ratio

    The function attempts to robustly map multiple possible source columns (given the schema's inconsistent descriptions).
    """
    df = df.copy()

    # --- Create Female (IV) ---
    if 'consumer_credit' in df.columns:
        f_raw = pd.to_numeric(df['consumer_credit'], errors='coerce')
        vals = set(f_raw.dropna().unique())
        if vals.issubset({0, 1}):
            df['Female'] = f_raw.astype('Int64')
        else:
            # treat values >= 0.5 as female
            df['Female'] = (f_raw >= 0.5).astype('Int64')
    elif 'female' in df.columns:
        f_raw = pd.to_numeric(df['female'], errors='coerce')
        vals = set(f_raw.dropna().unique())
        if vals.issubset({0, 1}):
            df['Female'] = f_raw.astype('Int64')
        else:
            df['Female'] = (f_raw >= 0.5).astype('Int64')
    else:
        raise KeyError("No gender column found. Expected 'consumer_credit' or 'female'.")

    # --- Create Approved (DV) ---
    if 'mortgage_credit' in df.columns:
        mc = pd.to_numeric(df['mortgage_credit'], errors='coerce')
        # mortgage_credit described as 1 = denied, 0 = accepted; we want Approved = 1 for accepted
        approved_cont = 1 - mc
        # Ensure binary: treat >= 0.5 as approved
        df['Approved'] = (approved_cont >= 0.5).astype('Int64')
    elif 'Unnamed: 0' in df.columns:
        u0 = pd.to_numeric(df['Unnamed: 0'], errors='coerce')
        vals = set(u0.dropna().unique())
        if vals.issubset({0, 1}):
            df['Approved'] = u0.astype('Int64')
        else:
            df['Approved'] = (u0 >= u0.median()).astype('Int64')
    elif 'accept' in df.columns:
        acc = pd.to_numeric(df['accept'], errors='coerce')
        df['Approved'] = (acc >= acc.median()).astype('Int64')
    else:
        raise KeyError("No approval/denial column found. Expected 'mortgage_credit', 'Unnamed: 0', or 'accept'.")

    # --- Map and standardize control variables (if present) ---
    # Preserve the required output order for control variables:
    mapping_list: List[tuple] = [
        ('black', 'Black'),
        ('bad_history', 'BadHistory'),
        ('loan_to_value', 'LoanToValue'),
        ('denied_PMI', 'DeniedPMI'),
        ('housing_expense_ratio', 'HousingExpenseRatio'),
        ('self_employed', 'SelfEmployed'),
        ('married', 'Married'),
        ('PI_ratio', 'PI_ratio')
    ]

    for src_col, out_col in mapping_list:
        if src_col in df.columns:
            df[out_col] = pd.to_numeric(df[src_col], errors='coerce')
        else:
            # create column with NaNs so downstream code can rely on consistent column names
            df[out_col] = np.nan

    # --- Drop rows missing DV or IV ---
    df = df.dropna(subset=['Approved', 'Female'])

    # --- For numeric controls, fill remaining missing values with the column median if possible ---
    control_cols = [out for (_, out) in mapping_list]
    for c in control_cols:
        # Ensure numeric type
        df[c] = pd.to_numeric(df[c], errors='coerce')
        if df[c].notna().any():
            median = df[c].median()
            df[c] = df[c].fillna(median)
        else:
            # leave as all-NaN (keeps column but indicates no information available)
            df[c] = df[c].astype(float)

    # Ensure final types for Approved and Female are integers (0/1)
    # Also defensively ensure they are strictly 0/1
    df['Approved'] = pd.to_numeric(df['Approved'], errors='coerce')
    df['Approved'] = (df['Approved'] >= 0.5).astype(int)

    df['Female'] = pd.to_numeric(df['Female'], errors='coerce')
    df['Female'] = (df['Female'] >= 0.5).astype(int)

    # Return only the columns needed for modeling (keeps order predictable)
    final_cols = ['Approved', 'Female'] + control_cols
    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Fit a logistic regression predicting Approved (1 = approved) from Female and controls.

    Returns a dictionary with keys:
      - 'result': the fitted statsmodels Logit or fallback result object
      - 'robust_result': an object that exposes robust-covariance params/pvalues/conf_int (HC1)
      - 'summary_table': a pandas DataFrame with odds ratios, 95% CI and p-values for coefficients

    The model includes a constant and uses robust (HC1) standard errors when reporting CIs/p-values.

    If the standard Logit fit fails due to singularity/perfect separation, falls back to GLM(Binomial).
    """
    df = df.copy()

    # Define regressors to include in the model. Keep the order stable.
    regressors = ['Female', 'Black', 'BadHistory', 'LoanToValue', 'DeniedPMI',
                  'HousingExpenseRatio', 'SelfEmployed', 'Married', 'PI_ratio']

    # Keep only regressors present in df (they should all be present as columns; if some are entirely NaN, they will be dropped by dropna)
    available_regs = [r for r in regressors if r in df.columns]

    # Drop rows with missing values in any of the model variables (DV or chosen regressors)
    model_cols = ['Approved'] + available_regs
    df_model = df[model_cols].dropna()

    if df_model.shape[0] == 0:
        raise ValueError('No rows available for modeling after dropping NaNs.')

    # Ensure dependent variable is binary 0/1
    y = pd.to_numeric(df_model['Approved'], errors='coerce')
    if not set(y.dropna().unique()).issubset({0, 1}):
        # defensively binarize by threshold
        y = (y >= 0.5).astype(int)
    else:
        y = y.astype(int)

    X = df_model[available_regs].copy()
    # Ensure regressors are numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression, with fallback to GLM if Logit encounters singularity/perfect separation
    result = None
    try:
        logit = sm.Logit(y, X)
        result = logit.fit(disp=False)
    except (np.linalg.LinAlgError, PerfectSeparationError):
        # Fallback: use GLM with Binomial family (IRLS) which is more robust to separation/singularity
        glm = sm.GLM(y, X, family=sm.families.Binomial())
        result = glm.fit(disp=False)

    # Compute HC1 robust covariance matrix for the fitted result
    try:
        cov_hc1 = sm.stats.sandwich_covariance.cov_hc1(result)
    except Exception:
        # If for some reason sandwich covariance fails, fall back to the model's default covariance
        cov_hc1 = result.cov_params()

    params = result.params
    # Ensure params is a pandas Series
    if not isinstance(params, pd.Series):
        params = pd.Series(params, index=X.columns)

    # Robust standard errors
    bse = pd.Series(np.sqrt(np.diag(cov_hc1)), index=params.index)

    # z-statistics and p-values (Wald tests) using normal approximation
    z_stats = params / bse
    pvalues = pd.Series(2 * (1 - stats.norm.cdf(np.abs(z_stats))), index=params.index)

    # 95% confidence intervals (using normal approximation)
    z_crit = stats.norm.ppf(1 - 0.05 / 2)
    conf_low = params - z_crit * bse
    conf_high = params + z_crit * bse
    conf_df = pd.DataFrame(np.column_stack([conf_low, conf_high]), index=params.index)

    # Build a robust_result object that provides commonly used attributes/methods
    def cov_params_func():
        return cov_hc1

    def conf_int_func():
        # Return DataFrame with two columns (like statsmodels' conf_int)
        return conf_df

    robust_result = SimpleNamespace(
        params=params,
        pvalues=pvalues,
        bse=bse,
        cov_params=cov_params_func,
        conf_int=conf_int_func
    )

    # Odds ratios and confidence intervals
    or_est = np.exp(params)
    conf_odds = np.exp(conf_df)

    summary_table = pd.DataFrame({
        'OR': or_est,
        '2.5%': conf_odds.iloc[:, 0],
        '97.5%': conf_odds.iloc[:, 1],
        'pvalue': pvalues
    })

    # Return both the fitted object and a neat summary table
    return {
        'result': result,
        'robust_result': robust_result,
        'summary_table': summary_table
    }