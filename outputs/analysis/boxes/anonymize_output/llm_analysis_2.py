from typing import Any
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm


# ======== TRANSFORM CODE ========
def _normalize_name(name: str) -> str:
    """Normalize column name for comparison: lowercase and remove non-alphanumeric."""
    return re.sub(r'[^0-9a-z]', '', str(name).lower())


def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Define desired final column names and plausible alternatives in raw data
    target_alternatives = {
        'Choice': ['feature1', 'choice', 'selected_option', 'option_chosen', 'response', 'selection'],
        'Gender': ['feature2', 'gender', 'sex'],
        'Age': ['feature3', 'age', 'age_years', 'years', 'child_age'],
        'DemoFirst': ['feature4', 'demofirst', 'demo_first', 'demonstrated_first', 'demoorder', 'demo_first_flag'],
        'SiteID': ['feature5', 'siteid', 'site_id', 'site', 'location']
    }

    # Build a lookup from normalized raw column names to actual column names
    col_lookup = { _normalize_name(col): col for col in df.columns }

    # Rename any recognized alternative columns to the exact target names required
    rename_map = {}
    for target, alts in target_alternatives.items():
        # If target already present, keep it
        if target in df.columns:
            continue
        # Otherwise look for alternatives
        for alt in alts:
            norm_alt = _normalize_name(alt)
            if norm_alt in col_lookup:
                rename_map[col_lookup[norm_alt]] = target
                break
    if rename_map:
        df = df.rename(columns=rename_map)

    # Coerce numeric columns where appropriate if they exist (helps dropna and comparisons)
    for col in ['Choice', 'Gender', 'Age', 'DemoFirst', 'SiteID']:
        if col in df.columns:
            if col == 'Age':
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    # NOTE: Do not aggressively drop rows here based on multiple raw columns.
    # Instead, construct the final variables and allow the model to decide which rows to use.
    # This preserves as many rows as possible for downstream handling.

    # Create the dependent variable: whether child chose the majority option
    # Interpret Choice == 2 as majority per original mapping; be robust to dtype
    if 'Choice' in df.columns:
        df['MajorityChoice'] = (df['Choice'] == 2).astype(int)
    else:
        df['MajorityChoice'] = np.nan

    # Recode gender to Male binary (1 = boy, 0 = girl)
    if 'Gender' in df.columns:
        df['Male'] = (df['Gender'] == 2).astype(int)
    else:
        df['Male'] = np.nan

    # Ensure DemoFirst is binary 0/1
    if 'DemoFirst' in df.columns:
        df['DemoFirst'] = (df['DemoFirst'] == 1).astype(int)
    else:
        df['DemoFirst'] = np.nan

    # Create Site as categorical string variable (keeps site identity for modeling)
    if 'SiteID' in df.columns:
        def _site_label(v):
            if pd.isna(v):
                return 'Site_missing'
            try:
                if float(v).is_integer():
                    return f"Site_{int(v)}"
            except Exception:
                pass
            return f"Site_{str(v)}"
        df['Site'] = df['SiteID'].apply(_site_label)
    elif 'Site' in df.columns:
        df['Site'] = df['Site'].astype(str)
    else:
        # Keep Site column present (filled with NaN) so model sees the column even if missing
        df['Site'] = np.nan

    # Center Age to improve interpretability and numerical stability
    if 'Age' in df.columns:
        df['Age'] = df['Age'].astype(float)
        age_mean = df['Age'].mean()
        # If mean is NaN (e.g., all ages missing), keep Age_c as NaN
        if pd.isna(age_mean):
            df['Age_c'] = np.nan
        else:
            df['Age_c'] = df['Age'] - age_mean
    else:
        df['Age'] = np.nan
        df['Age_c'] = np.nan
        age_mean = np.nan

    # Quadratic age term to capture non-linear developmental patterns
    df['Age_c_sq'] = df['Age_c'] ** 2

    # Final dataframe returned contains all columns used in the model
    # Keep only relevant columns (but do not drop others if user wants them)
    cols_needed = ['MajorityChoice', 'Age', 'Age_c', 'Age_c_sq', 'Male', 'DemoFirst', 'Site']
    for col in cols_needed:
        if col not in df.columns:
            df[col] = np.nan

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # Build design matrix for a binomial (logistic) regression predicting choosing the majority option.
    # The model includes: Age centered and quadratic, site fixed effects, interactions between Age_c and Site
    # (to allow age slopes to vary by site), and controls for gender and demonstration order.

    # Copy to avoid modifying original
    data = df.copy()

    # Ensure required columns are present (as columns; they may contain NaNs)
    required = ['MajorityChoice', 'Age_c', 'Age_c_sq', 'Male', 'DemoFirst', 'Site']
    for c in required:
        if c not in data.columns:
            raise ValueError(f"Required column {c} not found in dataframe")

    # Drop rows with missing values in the variables used for modeling
    model_subset = ['MajorityChoice', 'Age_c', 'Age_c_sq', 'Male', 'DemoFirst', 'Site']
    data_model = data.dropna(subset=model_subset)

    # If no observations remain, return a placeholder results object with appropriate attributes
    if data_model.shape[0] == 0:
        # Construct a placeholder object exposing common attributes used by downstream code
        class EmptyModelResult:
            def __init__(self, params_index):
                # params, bse, pvalues as pandas Series filled with NaN
                self.params = pd.Series(index=params_index, data=[np.nan] * len(params_index)).astype(float)
                self.bse = pd.Series(index=params_index, data=[np.nan] * len(params_index)).astype(float)
                self.pvalues = pd.Series(index=params_index, data=[np.nan] * len(params_index)).astype(float)
                self.nobs = 0
                self.aic = np.nan

            def summary(self):
                return "No observations available for modeling. EmptyModelResult with NaN parameters."

            def get_robustcov_results(self, cov_type='HC1'):
                return self

        # Define base predictor names as used in the modeling code
        base_predictors = ['const', 'Age_c', 'Age_c_sq', 'Male', 'DemoFirst']
        # In absence of any site levels, we do not add site dummies or interactions
        return EmptyModelResult(base_predictors)

    # Use data_model for all subsequent steps
    data = data_model

    # Create site dummy variables (drop first to avoid multicollinearity)
    site_dummies = pd.get_dummies(data['Site'], prefix='Site', drop_first=True)

    # Base predictors (ensure numeric dtype)
    X = data[['Age_c', 'Age_c_sq', 'Male', 'DemoFirst']].astype(float).copy()

    # Add site dummies (site main effects) if any
    if not site_dummies.empty:
        X = pd.concat([X, site_dummies.astype(float)], axis=1)

    # Add interactions between Age_c and each site dummy to allow site-specific age slopes
    if not site_dummies.empty:
        for col in site_dummies.columns:
            inter_name = f"{col}_x_Age"
            X[inter_name] = site_dummies[col].astype(float) * data['Age_c'].astype(float)

    # Ensure there is at least one predictor column before adding constant
    if X.shape[1] == 0:
        # This should not normally happen because Age_c etc. are required, but guard just in case.
        raise ValueError("No predictor columns available for modeling after constructing design matrix.")

    # Add intercept
    X = sm.add_constant(X, has_constant='add')

    # Outcome
    y = data['MajorityChoice'].astype(int)

    # Fit logistic regression (GLM with binomial family). Fit and then obtain robust covariance (HC1) for SEs.
    model_glm = sm.GLM(y, X, family=sm.families.Binomial())
    fit_res = model_glm.fit()
    try:
        results = fit_res.get_robustcov_results(cov_type='HC1')
    except Exception:
        # Fallback: if robustcov method unavailable, return the original fit
        results = fit_res

    # Return the fitted results object (has summary(), params, pvalues, etc.)
    return results