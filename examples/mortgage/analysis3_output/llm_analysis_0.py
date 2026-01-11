from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/examples/mortgage/analysis3_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw DataFrame into the analysis-ready DataFrame.

    Produces these final columns used in the model:
      - Approved: binary (1 = accepted/approved, 0 = denied)
      - Female: binary (1 = female, 0 = male)
      - BadHistory, Married, SelfEmployed, LoanToValue: control variables copied from originals
      - PI_ratio_z, Denied_PMI_z, HousingExpenseRatio_z, CreditScore_z: standardized continuous controls

    Rules implemented:
      - Prefer 'mortgage_credit' as the denial indicator (schema: 1 = denied, 0 = accepted). If not available, try 'Unnamed: 0'.
      - Prefer 'consumer_credit' as the gender indicator (schema: 1 = female, 0 = male). If not available, try 'female' (thresholded at >0.5).
      - Coerce relevant columns to numeric and drop rows with missing values in any final variables.
    """
    df = df.copy()

    # Ensure numeric for many candidate columns (coerce errors to NaN)
    candidate_cols = ['mortgage_credit', 'consumer_credit', 'bad_history', 'married', 'PI_ratio',
                      'housing_expense_ratio', 'accept', 'loan_to_value', 'denied_PMI',
                      'female', 'self_employed', 'Unnamed: 0']
    for c in candidate_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Determine mortgage denial column (expected coding: 1 = denied, 0 = accepted)
    if 'mortgage_credit' in df.columns:
        df['mortgage_credit_bin'] = df['mortgage_credit']
    elif 'Unnamed: 0' in df.columns:
        df['mortgage_credit_bin'] = df['Unnamed: 0']
    else:
        raise ValueError("No mortgage acceptance/denial column found in dataset. Expected 'mortgage_credit' or 'Unnamed: 0'.")

    # Create Approved indicator: 1 = accepted (mortgage_credit_bin == 0), 0 = denied
    # Use equality check to ensure we only set Approved when value explicitly equals 0 or 1
    df['Approved'] = df['mortgage_credit_bin'].apply(lambda x: 1 if pd.notnull(x) and x == 0 else (0 if pd.notnull(x) and x == 1 else np.nan))

    # Gender: prefer consumer_credit (schema: 1 = female, 0 = male)
    if 'consumer_credit' in df.columns:
        df['Female'] = df['consumer_credit'].apply(lambda x: int(x) if pd.notnull(x) else np.nan)
    elif 'female' in df.columns:
        # 'female' column in this dataset appears non-binary / continuous in schema; threshold to produce indicator
        df['Female'] = df['female'].apply(lambda x: 1 if pd.notnull(x) and x > 0.5 else (0 if pd.notnull(x) else np.nan))
    else:
        raise ValueError("No gender column found. Expected 'consumer_credit' or 'female'.")

    # Map other controls (copy and coerce to numeric if present)
    df['BadHistory'] = df['bad_history'] if 'bad_history' in df.columns else np.nan
    df['Married'] = df['married'] if 'married' in df.columns else np.nan
    df['SelfEmployed'] = df['self_employed'] if 'self_employed' in df.columns else np.nan
    df['LoanToValue'] = df['loan_to_value'] if 'loan_to_value' in df.columns else np.nan
    # Copy raw continuous controls, then standardize below
    df['PI_ratio_raw'] = df['PI_ratio'] if 'PI_ratio' in df.columns else np.nan
    df['Denied_PMI_raw'] = df['denied_PMI'] if 'denied_PMI' in df.columns else np.nan
    df['HousingExpenseRatio_raw'] = df['housing_expense_ratio'] if 'housing_expense_ratio' in df.columns else np.nan
    df['CreditScore_raw'] = df['accept'] if 'accept' in df.columns else np.nan

    # Standardize continuous controls to z-scores (mean 0, sd 1) when possible.
    # Use ddof=0 (population sd) to avoid dividing by zero in small samples; leave NaN when column is all NaN.
    def zscore(series: pd.Series) -> pd.Series:
        if series.dropna().shape[0] == 0:
            return pd.Series(index=series.index, dtype=float)
        mu = series.mean()
        sigma = series.std(ddof=0)
        if sigma == 0 or np.isnan(sigma):
            return (series - mu).astype(float)
        return (series - mu) / sigma

    df['PI_ratio_z'] = zscore(df['PI_ratio_raw'])
    df['Denied_PMI_z'] = zscore(df['Denied_PMI_raw'])
    df['HousingExpenseRatio_z'] = zscore(df['HousingExpenseRatio_raw'])
    df['CreditScore_z'] = zscore(df['CreditScore_raw'])

    # Select final columns to ensure they exist and drop rows with missing values in any of them
    final_cols = ['Approved', 'Female', 'BadHistory', 'Married', 'SelfEmployed',
                  'PI_ratio_z', 'Denied_PMI_z', 'LoanToValue', 'HousingExpenseRatio_z', 'CreditScore_z']

    # If any of the named control columns do not exist in the DataFrame, they will be NaN; drop rows with NaNs in final columns
    df_final = df.copy()
    df_final = df_final.dropna(subset=final_cols)

    # Ensure final columns have appropriate dtypes
    # Approved and Female should be integer indicators
    df_final['Approved'] = df_final['Approved'].astype(int)
    df_final['Female'] = df_final['Female'].astype(int)
    # Other controls as floats
    df_final['BadHistory'] = df_final['BadHistory'].astype(float)
    df_final['Married'] = df_final['Married'].astype(float)
    df_final['SelfEmployed'] = df_final['SelfEmployed'].astype(float)
    df_final['LoanToValue'] = df_final['LoanToValue'].astype(float)

    # Return the transformed DataFrame (preserve extras but guarantee final_cols exist)
    return df_final


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression predicting Approved (1 = accepted) from Female and controls.

    Model specification (primary):
      logit( P(Approved=1) ) = beta0 + beta1 * Female + beta2 * BadHistory + beta3 * Married +
                              beta4 * SelfEmployed + beta5 * PI_ratio_z + beta6 * Denied_PMI_z +
                              beta7 * LoanToValue + beta8 * HousingExpenseRatio_z + beta9 * CreditScore_z

    Returns the fitted statsmodels Logit result object (or a closely compatible fallback).
    """
    # Ensure required columns exist
    required = ['Approved', 'Female', 'BadHistory', 'Married', 'SelfEmployed',
                'PI_ratio_z', 'Denied_PMI_z', 'LoanToValue', 'HousingExpenseRatio_z', 'CreditScore_z']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Build design matrix X and outcome y
    X = df[['Female', 'BadHistory', 'Married', 'SelfEmployed',
            'PI_ratio_z', 'Denied_PMI_z', 'LoanToValue', 'HousingExpenseRatio_z', 'CreditScore_z']].astype(float).copy()
    X = sm.add_constant(X, has_constant='add')
    y = df['Approved'].astype(int)

    # If any predictor column has zero variance (or a single unique value), add tiny jitter to avoid singular matrix.
    # The jitter is extremely small and deterministic (seeded) so as not to materially change the data but to
    # allow matrix inversion when a column is constant in the sample.
    rng = np.random.RandomState(0)
    for col in X.columns:
        if col == 'const' or col == 'const' or col == 'const':  # harmless repeated check to keep const name stable
            # statsmodels uses 'const' as default name for intercept
            if col == 'const':
                continue
        # Consider a column constant if std (ddof=0) is effectively zero or there is <=1 unique value (ignoring NaN)
        col_non_na = X[col].dropna()
        if col_non_na.shape[0] == 0:
            continue
        if np.isclose(col_non_na.std(ddof=0), 0.0) or col_non_na.nunique(dropna=True) <= 1:
            # add tiny deterministic noise
            noise = rng.normal(loc=0.0, scale=1e-8, size=X.shape[0])
            X[col] = X[col] + noise

    # Fit logistic regression (maximum likelihood). Use try/except to surface convergence problems.
    try:
        logit_model = sm.Logit(y, X)
        results = logit_model.fit(disp=False)
    except Exception as e:
        # If standard Logit fails (e.g., singular matrix / perfect separation), attempt a regularized fit as a fallback.
        # Regularized fit should succeed in many degenerate cases by stabilizing estimation.
        try:
            logit_model = sm.Logit(y, X)
            # Try a small L2-style regularization via fit_regularized. Statsmodels' fit_regularized may accept different
            # arguments depending on version; use a small alpha to minimally regularize.
            results = logit_model.fit_regularized(method='l1', alpha=1e-6, disp=False)
        except Exception:
            try:
                # If the above fails, try a slightly larger penalty
                results = logit_model.fit_regularized(method='l1', alpha=1e-4, disp=False)
            except Exception:
                # As a last resort, re-raise the original exception with context.
                raise RuntimeError(f"Logit model failed to fit: {e}")

    return results


if __name__ == "__main__":
    # Example run (will execute when this file is run directly)
    transformed = transform(df)
    fitted = model(transformed)
    # Print a brief summary if the result object supports it
    try:
        print(fitted.summary())
    except Exception:
        # Fallback: print parameters if available
        try:
            print("Params:", fitted.params)
        except Exception:
            print("Model fitted; result object type:", type(fitted))