from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/shuffle_names_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # --- Build female indicator ---
    # Prefer a clear 0/1 column indicating female (dataset 'consumer_credit' sometimes encodes female)
    if 'consumer_credit' in df.columns:
        # Coerce to numeric first (preserve NaNs)
        cc = pd.to_numeric(df['consumer_credit'], errors='coerce')
        non_na = cc.dropna()
        # If non-empty and looks binary (0/1), use it as female indicator (keep NaN where present)
        if not non_na.empty and set(np.unique(non_na)) .issubset({0, 1}):
            df['female'] = cc.astype(float)
        else:
            # fallback: threshold at 0.5, but preserve NaNs
            female_vals = pd.Series(index=cc.index, dtype='float')
            mask = cc.notna()
            female_vals.loc[mask] = (cc.loc[mask] >= 0.5).astype(float)
            female_vals.loc[~mask] = np.nan
            df['female'] = female_vals
    elif 'female' in df.columns:
        # Column exists but may be continuous probabilities; threshold at 0.5, preserving NaNs
        ff = pd.to_numeric(df['female'], errors='coerce')
        female_vals = pd.Series(index=ff.index, dtype='float')
        mask = ff.notna()
        female_vals.loc[mask] = (ff.loc[mask] >= 0.5).astype(float)
        female_vals.loc[~mask] = np.nan
        df['female'] = female_vals
    else:
        # If neither column exists, create NA column to fail gracefully
        df['female'] = np.nan

    # --- Build approved outcome ---
    # Priority: 'Unnamed: 0' (appears in this dataset as acceptance flag), then 'mortgage_credit' (sometimes 1=denied), then 'accept' (ordinal) as fallback.
    approved = pd.Series(index=df.index, dtype='float')
    if 'Unnamed: 0' in df.columns:
        uniq = df['Unnamed: 0'].dropna().unique()
        if set(np.unique(uniq)).issubset({0, 1}):
            approved = pd.to_numeric(df['Unnamed: 0'], errors='coerce').astype(float)
    if approved.isna().all() and 'mortgage_credit' in df.columns:
        # Many descriptions indicate mortgage_credit==1 means denied; convert to approved = 1 - mortgage_credit
        mc = pd.to_numeric(df['mortgage_credit'], errors='coerce')
        uniq = mc.dropna().unique()
        if set(np.unique(uniq)).issubset({0, 1}):
            approved = (1 - mc).astype(float)
    if approved.isna().all() and 'accept' in df.columns:
        # 'accept' may be acceptance quality on 1-6. Use threshold >=4 as accepted (reasonable fallback).
        acc = pd.to_numeric(df['accept'], errors='coerce')
        approved = pd.Series(index=acc.index, dtype='float')
        mask = acc.notna()
        approved.loc[mask] = (acc.loc[mask] >= 4).astype(float)
        approved.loc[~mask] = np.nan

    # If still all NA, create column of NA
    df['approved'] = approved

    # --- Controls: ensure columns exist; if not, create NA placeholders ---
    control_cols = ['PI_ratio', 'loan_to_value', 'housing_expense_ratio', 'denied_PMI',
                    'self_employed', 'married', 'bad_history', 'black']
    for col in control_cols:
        if col not in df.columns:
            df[col] = np.nan

    # --- Convert to numeric where possible ---
    for col in ['PI_ratio', 'loan_to_value', 'housing_expense_ratio', 'denied_PMI', 'black']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in ['self_employed', 'married', 'bad_history']:
        # treat as binary indicators when possible
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # If not strictly 0/1 among non-missing values, coerce nonzero to 1
        non_na = df[col].dropna()
        if not non_na.empty and not non_na.isin([0, 1]).all():
            mask = df[col].notna()
            df.loc[mask, col] = (df.loc[mask, col] != 0).astype(int)
        # keep as float to allow NaN
        df[col] = df[col].astype(float)

    # --- Impute continuous controls with median and create missing indicators, then standardize (z-score) ---
    cont_to_standardize = ['PI_ratio', 'loan_to_value', 'housing_expense_ratio', 'denied_PMI', 'black']
    for col in cont_to_standardize:
        miss_col = f"{col}_miss"
        z_col = f"{col}_z"
        df[miss_col] = df[col].isna().astype(int)
        median = df[col].median(skipna=True)
        # If median is NaN (entire column NA), fill with 0
        if pd.isna(median):
            median = 0.0
        df[col] = df[col].fillna(median)
        std = df[col].std(ddof=0)
        if pd.isna(std) or std == 0:
            # If zero variance, z-score will be zero
            df[z_col] = 0.0
        else:
            df[z_col] = (df[col] - median) / std

    # Ensure binary controls are filled (treat missing as 0 and mark missing)
    for col in ['self_employed', 'married', 'bad_history']:
        miss_col = f"{col}_miss"
        df[miss_col] = df[col].isna().astype(int)
        df[col] = df[col].fillna(0).astype(int)

    # --- Final filtering: drop rows missing outcome or gender ---
    df = df[~df['approved'].isna()]
    df = df[~df['female'].isna()]

    # Convert final binary columns to int
    df['approved'] = df['approved'].astype(int)
    df['female'] = df['female'].astype(int)

    # Keep only the columns needed for modeling to make the output tidy
    final_cols = [
        'approved', 'female',
        'PI_ratio_z', 'PI_ratio_miss',
        'loan_to_value_z', 'loan_to_value_miss',
        'housing_expense_ratio_z', 'housing_expense_ratio_miss',
        'denied_PMI_z', 'denied_PMI_miss',
        'self_employed', 'self_employed_miss',
        'married', 'married_miss',
        'bad_history', 'bad_history_miss',
        'black_z', 'black_miss'
    ]

    # Ensure all final columns exist (some may not have been created if original cols were absent)
    for col in final_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # df is expected to be the output of transform()
    # Build design matrix for logistic regression
    df = df.copy()

    # Define outcome and predictors
    y = df['approved']

    # predictors: female + controls
    predictors = [
        'female',
        'PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z', 'denied_PMI_z',
        'self_employed', 'married', 'bad_history', 'black_z'
    ]

    X = df[predictors]

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Drop rows with any remaining NA in X or y
    valid = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid]
    y = y.loc[valid]

    # Fit logistic regression (maximum likelihood)
    try:
        logit_model = sm.Logit(y, X)
        fit_res = logit_model.fit(disp=False)
    except Exception:
        # If convergence problems arise, try using a small regularization (statsmodels discrete doesn't support L2 in Logit directly)
        # As a fallback, use sklearn's LogisticRegression with liblinear solver
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000)
        lr.fit(X, y)
        # Build a minimal result-like object
        class SklearnResult:
            def __init__(self, model, X, y, predictors):
                self.model = model
                self.X = X
                self.y = y
                self.predictors = predictors
            def summary(self):
                coefs = np.concatenate(([self.model.intercept_[0]], self.model.coef_.ravel()))
                return f"Sklearn LogisticRegression fitted. Coefs (const + predictors): {coefs}"
        fit_res = SklearnResult(lr, X, y, predictors)

        return {
            'result_obj': fit_res,
            'method': 'sklearn_fallback'
        }

    # Compute odds ratios and 95% confidence intervals
    params = fit_res.params
    conf = fit_res.conf_int()
    odds_ratios = pd.DataFrame({
        'coef': params,
        'odds_ratio': np.exp(params),
        'ci_lower': np.exp(conf[0]),
        'ci_upper': np.exp(conf[1])
    })

    # Compute average marginal effects
    try:
        marg = fit_res.get_margeff(at='overall', method='dydx')
        marg_summary = marg.summary().as_text()
    except Exception:
        marg = None
        marg_summary = None

    # Package results
    results = {
        'model': fit_res,
        'odds_ratios': odds_ratios,
        'marginal_effects_summary': marg_summary,
        'design_matrix_shape': X.shape,
        'n_obs': int(y.shape[0])
    }

    return results