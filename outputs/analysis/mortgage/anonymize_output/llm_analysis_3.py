from typing import Any
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Optional: load a sample dataframe if running script; harmless if file missing in other contexts.
try:
    df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/anonymize_output/mortgage.csv')
except Exception:
    df = pd.DataFrame()


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Mapping from raw feature names (expected in some datasets) to
    # standardized raw column names we will work with.
    # Note: 'Denied' is included because some datasets may report denial
    # instead of acceptance; we'll convert if needed.
    rename_map = {
        'feature2': 'Female',            # 1 if female, 0 if male
        'feature3': 'Black',             # 1 if Black, 0 otherwise
        'feature5': 'SelfEmployed',      # 1 if self-employed
        'feature6': 'Married',           # 1 if married
        'feature7': 'MortgageScore',     # mortgage credit score (1-4)
        'feature8': 'ConsumerScore',     # consumer credit score (1-6)
        'feature9': 'BadCreditHistory',  # 1 if history of bad credit
        'feature10': 'DebtToIncome',     # total debt payments to income ratio
        'feature11': 'Denied',           # 1 if denied, 0 if accepted (sometimes present)
        'feature12': 'LTV',              # loan amount to appraised value ratio
        'feature14': 'Accepted'          # 1 if accepted, 0 if denied
    }

    # Desired raw columns we need (before standardizing continuous variables)
    desired_raw_cols = [
        'Female', 'Accepted', 'Black', 'SelfEmployed', 'Married',
        'BadCreditHistory', 'MortgageScore', 'ConsumerScore', 'DebtToIncome', 'LTV'
    ]

    # Helper to normalize column names for fuzzy matching
    def _normalize(name: str) -> str:
        return re.sub(r'[^0-9a-zA-Z]', '', str(name)).lower()

    # Build a lookup of normalized existing dataframe columns for quick matching
    normalized_cols = { _normalize(c): c for c in df.columns }

    # Ensure raw columns exist in the dataframe by copying from known source names if present,
    # or by finding fuzzy matches. If none found, create the column filled with NA.
    for raw_name in set(list(rename_map.values()) + desired_raw_cols):  # include 'Denied' as well
        if raw_name in df.columns:
            continue  # already present
        # Check if any of the expected source keys exists in the dataframe
        source_found = False
        for src, tgt in rename_map.items():
            if tgt != raw_name:
                continue
            # direct source name present?
            if src in df.columns:
                df[raw_name] = df[src]
                source_found = True
                break
            # fuzzy match source name
            norm_src = _normalize(src)
            if norm_src in normalized_cols:
                df[raw_name] = df[normalized_cols[norm_src]]
                source_found = True
                break
        if source_found:
            continue
        # Fallback: try to find any dataframe column that matches the raw_name by normalization
        norm_raw = _normalize(raw_name)
        if norm_raw in normalized_cols:
            df[raw_name] = df[normalized_cols[norm_raw]]
            continue
        # No source found: create the column with NA values
        df[raw_name] = pd.NA

    # If Accepted is missing but Denied exists, create Accepted = 1 - Denied
    if ('Accepted' not in df.columns or df['Accepted'].isna().all()) and 'Denied' in df.columns:
        # Convert Denied to numeric, then compute Accepted
        df['Denied'] = pd.to_numeric(df['Denied'], errors='coerce')
        df['Accepted'] = 1 - df['Denied']

    # Now convert raw columns to numeric types (safe casting)
    for c in desired_raw_cols:
        # Ensure column exists (we created placeholders above)
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows with missing values in any of the essential raw modeling columns
    df = df.dropna(subset=desired_raw_cols)

    # Make sure binary columns are 0/1 integers
    for bcol in ['Female', 'Black', 'SelfEmployed', 'Married', 'BadCreditHistory', 'Accepted']:
        # After dropna above, these should be non-missing numeric values; coerce and cast to int safely.
        if bcol in df.columns and not df.empty:
            # Use round then astype(int) to avoid float representation issues like 1.0/0.0
            df[bcol] = pd.to_numeric(df[bcol], errors='coerce').round().astype(int)

    # Standardize continuous predictors (z-scores) for better numeric stability in modeling
    # Use population std (ddof=0) for consistent scaling
    cont_cols = {
        'MortgageScore': 'MortgageScore_z',
        'ConsumerScore': 'ConsumerScore_z',
        'DebtToIncome': 'DebtToIncome_z',
        'LTV': 'LTV_z'
    }
    for orig, zname in cont_cols.items():
        # orig columns have been created and coerced to numeric above
        # If orig not present create as NA (shouldn't happen)
        if orig not in df.columns:
            df[orig] = pd.NA
        # compute mean/std on available data
        if df.empty:
            df[zname] = pd.Series(dtype=float)
            continue
        mean = df[orig].mean()
        std = df[orig].std(ddof=0)
        # If std is 0 (degenerate), create zero column to avoid division by zero
        if std == 0 or pd.isna(std):
            df[zname] = 0.0
        else:
            df[zname] = (df[orig] - mean) / std

    # Keep only the columns needed for modeling to avoid accidental use of raw features later
    keep_cols = ['Female', 'Accepted', 'Black', 'SelfEmployed', 'Married', 'BadCreditHistory',
                 'MortgageScore_z', 'ConsumerScore_z', 'DebtToIncome_z', 'LTV_z']
    # Ensure all keep_cols exist (they should); if not, create to avoid KeyError (but they should exist)
    for kc in keep_cols:
        if kc not in df.columns:
            df[kc] = pd.NA

    final_df = df[keep_cols].copy()

    return final_df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # Logistic regression: probability(Accepted = 1) modeled with applicant gender and controls
    # We use statsmodels Logit and return robust standard errors (HC3).

    # Define predictors (main specification). Female is the variable of interest.
    X_cols = [
        'Female',
        'Black',
        'SelfEmployed',
        'Married',
        'BadCreditHistory',
        'MortgageScore_z',
        'ConsumerScore_z',
        'DebtToIncome_z',
        'LTV_z'
    ]

    # Quick checks to avoid passing empty arrays to statsmodels
    if df is None:
        return None
    if not isinstance(df, pd.DataFrame):
        return None
    # Ensure required columns exist
    required = set(X_cols + ['Accepted'])
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        # Missing required columns; cannot proceed
        return None

    # Drop rows with missing values in these model columns as a final safeguard
    model_df = df[list(required)].dropna()
    if model_df.shape[0] == 0:
        # No data to fit the model
        return None

    # Prepare design matrix
    X = model_df[X_cols].astype(float)
    # add constant
    X = sm.add_constant(X, has_constant='add')
    y = model_df['Accepted'].astype(float)

    # As an additional safeguard, ensure X has at least one row and one column
    if X.size == 0 or X.shape[0] == 0 or X.shape[1] == 0:
        return None

    # Fit logistic regression
    logit = sm.Logit(y, X)
    try:
        res = logit.fit(disp=False)
    except Exception:
        # If convergence issues arise, try a small regularization by using GLM with binomial
        glm = sm.GLM(y, X, family=sm.families.Binomial())
        res = glm.fit()

    # Return results with robust (HC3) covariance for inference
    try:
        robust_res = res.get_robustcov_results(cov_type='HC3')
    except Exception:
        # If not available (e.g., GLM results), return the original fit
        robust_res = res

    return robust_res