from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/.venv/lib/python3.10/site-packages/blade_bench/datasets/compas/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare COMPAS dataset for modeling.
    - Drops rows missing core fields needed for modeling.
    - Creates a categorical race_group and binary indicators Race_Black and Race_White.
    - Ensures numeric columns used as features are numeric.

    Returns modified dataframe that includes all columns used in later modeling:
      - Race_Black, Race_White, race_group
      - all original feature columns (priors_count, age, c_jail_time, juv_fel_count, juv_other_count,
        juv_misd_count, c_charge_degree:F, sex:Male, days_b_screening_arrest, two_year_recid)
    """
    # Copy to avoid modifying original in-place
    df = df.copy()

    # Ensure columns exist; if some expected columns are missing, this will raise KeyError
    required_cols = [
        'two_year_recid', 'race:African-American', 'race:Caucasian',
        'priors_count', 'age', 'c_jail_time', 'juv_fel_count', 'juv_other_count',
        'juv_misd_count', 'c_charge_degree:F', 'sex:Male', 'days_b_screening_arrest'
    ]

    # Drop rows with missing critical variables
    df = df.dropna(subset=required_cols)

    # Convert numeric columns to numeric types (coerce errors -> NaN, then drop any newly created NaNs)
    num_cols = ['priors_count', 'age', 'c_jail_time', 'juv_fel_count', 'juv_other_count',
                'juv_misd_count', 'days_b_screening_arrest']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=num_cols)

    # Create race_group and binary indicators. The dataset encodes races as one-hot columns.
    def _race_group(row):
        if row.get('race:African-American', 0) == 1:
            return 'African-American'
        elif row.get('race:Caucasian', 0) == 1:
            return 'Caucasian'
        elif row.get('race:Hispanic', 0) == 1:
            return 'Hispanic'
        elif row.get('race:Asian', 0) == 1:
            return 'Asian'
        elif row.get('race:Native_American', 0) == 1:
            return 'Native_American'
        elif row.get('race:Other', 0) == 1:
            return 'Other'
        else:
            return 'Unknown'

    df['race_group'] = df.apply(_race_group, axis=1)

    # Create indicator columns used in models
    df['Race_Black'] = (df['race_group'] == 'African-American').astype(int)
    df['Race_White'] = (df['race_group'] == 'Caucasian').astype(int)

    # Ensure charge-degree and sex dummies are numeric 0/1
    df['c_charge_degree:F'] = pd.to_numeric(df['c_charge_degree:F'], errors='coerce').fillna(0).astype(int)
    df['sex:Male'] = pd.to_numeric(df['sex:Male'], errors='coerce').fillna(0).astype(int)

    # Keep only rows where race is known as African-American or Caucasian for the main comparison
    df = df[df['race_group'].isin(['African-American', 'Caucasian'])].copy()

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    1) Trains a proxy COMPAS logistic model to predict two_year_recid using criminal history
       and demographic features but WITHOUT race. The predicted probability is used as
       ProxyCompasScore (continuous 0-1) and is the primary dependent variable for fairness tests.

    2) Tests whether Race_Black predicts higher ProxyCompasScore after controlling for the
       same features used in the proxy model (linear regression with robust SEs).

    3) Additionally creates a binary high-risk label from ProxyCompasScore and fits a logistic
       regression testing whether Race_Black predicts being classified as high-risk controlling
       for the same covariates.

    Returns a dictionary with fitted model results for inspection.
    """
    import statsmodels.api as sm
    import numpy as np

    # Work on a copy
    df = df.copy()

    # Define features to use for the proxy model (explicitly exclude any race variables)
    feature_cols = [
        'priors_count', 'age', 'c_jail_time', 'juv_fel_count', 'juv_other_count',
        'juv_misd_count', 'c_charge_degree:F', 'sex:Male', 'days_b_screening_arrest'
    ]

    # Ensure no missing values in the modeling subset
    model_df = df.dropna(subset=feature_cols + ['two_year_recid', 'Race_Black']).copy()

    # Prepare X and y for the proxy logistic model (WITHOUT race)
    X = model_df[feature_cols].astype(float)
    X = sm.add_constant(X)
    y = model_df['two_year_recid'].astype(int)

    # Fit logistic regression (proxy COMPAS) using statsmodels for interpretability
    # Use a try/except because Logit may sometimes be unstable; fall back to glm if needed
    try:
        proxy_logit = sm.Logit(y, X).fit(disp=False)
    except Exception:
        proxy_logit = sm.GLM(y, X, family=sm.families.Binomial()).fit()

    # Predicted probabilities (proxy score)
    model_df['ProxyCompasScore'] = proxy_logit.predict(X)

    # Add the score back to the input dataframe (align by index)
    df = df.merge(model_df[['ProxyCompasScore']], left_index=True, right_index=True, how='left')

    # -------------------------
    # Test 1: Does Race_Black predict higher ProxyCompasScore controlling for features?
    # Linear model: ProxyCompasScore ~ Race_Black + controls
    # -------------------------
    # Prepare data for OLS
    ols_cols = ['ProxyCompasScore', 'Race_Black'] + feature_cols
    ols_df = model_df[['ProxyCompasScore', 'Race_Black'] + feature_cols].dropna().copy()
    X_ols = ols_df[['Race_Black'] + feature_cols].astype(float)
    X_ols = sm.add_constant(X_ols)
    y_ols = ols_df['ProxyCompasScore'].astype(float)

    ols_model = sm.OLS(y_ols, X_ols).fit(cov_type='HC3')

    # -------------------------
    # Test 2: Binary classification test
    # Create a high-risk label from the ProxyCompasScore and test whether Race_Black predicts it
    # We'll use the 0.5 threshold and also provide an option to use the median as a robustness check.
    # -------------------------
    # Use 0.5 threshold; if scores are skewed, user can inspect distribution and rerun with a different threshold
    threshold = 0.5
    model_df['ProxyHighRisk'] = (model_df['ProxyCompasScore'] >= threshold).astype(int)

    # Fit logistic regression: ProxyHighRisk ~ Race_Black + controls
    X_clf = model_df[['Race_Black'] + feature_cols].astype(float)
    X_clf = sm.add_constant(X_clf)
    y_clf = model_df['ProxyHighRisk'].astype(int)

    try:
        clf_logit = sm.Logit(y_clf, X_clf).fit(disp=False)
    except Exception:
        clf_logit = sm.GLM(y_clf, X_clf, family=sm.families.Binomial()).fit()

    # Package results. Return model objects so the caller can inspect summaries, coefficients, p-values.
    results = {
        'proxy_model': proxy_logit,                # Fitted model used to generate ProxyCompasScore
        'fairness_score_ols': ols_model,           # OLS test of continuous score bias (Race_Black coefficient is key)
        'fairness_classif_logit': clf_logit,       # Logistic test for being classified as high-risk
        'transformed_df_with_score': model_df      # Dataframe used for the fairness tests (contains ProxyCompasScore and ProxyHighRisk)
    }

    return results


