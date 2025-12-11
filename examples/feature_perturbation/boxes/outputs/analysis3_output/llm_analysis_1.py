from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset to the analysis dataframe.

    Produces these final columns used in modeling:
      - ChoseMajority: binary DV (1 if child chose majority option, 0 otherwise)
      - AgeC: centered age in years (continuous IV)
      - SiteID: string categorical site identifier (IV; used with C(SiteID) in formula)
      - Gender: categorical gender (values 'girl' or 'boy') used as control
      - MajorityShownFirst: binary indicator (0/1) whether majority was demonstrated first (control)

    Notes about the raw schema: the provided field names are slightly inconsistent in their descriptions.
    According to the provided schema, the 'culture' column contains the child's age in years (4-14),
    the 'age' column is a 0/1 indicator for whether the majority option was demonstrated first,
    and 'y' is the site ID. We follow that mapping below.
    """
    df = df.copy()

    # Required raw columns (as referenced in the mapping above)
    required_cols = ['majority_first', 'culture', 'age', 'y', 'gender']
    df = df.dropna(subset=required_cols)

    # Dependent variable: 1 if child chose the majority option (original code 2), else 0
    df['ChoseMajority'] = (df['majority_first'] == 2).astype(int)

    # Age in years: 'culture' per schema
    df['AgeYears'] = pd.to_numeric(df['culture'], errors='coerce')

    # Centered age
    # If all ages are missing or constant, this will produce zeros or NaNs; downstream dropna will handle NaNs
    if df['AgeYears'].notna().any():
        df['AgeC'] = df['AgeYears'] - df['AgeYears'].mean()
    else:
        df['AgeC'] = np.nan

    # SiteID: use 'y' as site id, convert to string categorical
    # Use safe string conversion (don't cast to int which may fail)
    df['SiteID'] = 'site_' + df['y'].astype(str)

    # Gender mapping: 1=girl, 2=boy; fallback to string of original code if unexpected
    df['Gender'] = df['gender'].map({1: 'girl', 2: 'boy'})
    df.loc[df['Gender'].isna(), 'Gender'] = df.loc[df['Gender'].isna(), 'gender'].astype(str)
    df['Gender'] = df['Gender'].astype('category')

    # MajorityShownFirst: 'age' per schema encodes this 0/1 indicator
    df['MajorityShownFirst'] = pd.to_numeric(df['age'], errors='coerce').fillna(0).astype(int)
    df['MajorityShownFirst'] = df['MajorityShownFirst'].clip(0, 1)

    # Final drop: ensure all final columns have no missing values
    final_cols = ['ChoseMajority', 'AgeC', 'SiteID', 'Gender', 'MajorityShownFirst']
    df = df.dropna(subset=final_cols)

    # Reset index for reproducibility
    df = df.reset_index(drop=True)

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression to predict probability of choosing the majority option.

    Model specification:
      - DV: ChoseMajority (binary)
      - IVs: Age (centered) and Site (categorical); include their interaction to test whether age trajectories differ across sites
      - Controls: Gender (categorical), MajorityShownFirst (binary)

    We attempt to fit a binomial logit via statsmodels' formula interface. If maximum-likelihood
    optimization fails due to singularities (e.g., perfect collinearity or separation), we fall back
    to a regularized GLM (binomial) to obtain stable coefficients. We then attempt to compute
    cluster-robust standard errors clustered by SiteID; if that fails we return the fitted result object.
    """
    formula = 'ChoseMajority ~ AgeC * C(SiteID) + C(Gender) + MajorityShownFirst'

    # First attempt: standard logit (MLE)
    try:
        model_fit = smf.logit(formula=formula, data=df).fit(disp=False)
    except Exception as e:
        # If logit MLE fails (e.g., singular Hessian / perfect collinearity / separation), fall back to regularized GLM
        # Use a very small L2 penalty to stabilize estimation while preserving interpretation
        try:
            glm_model = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
            # small alpha, L1_wt=0 for pure L2
            model_fit = glm_model.fit_regularized(method='lbfgs', alpha=1e-6, L1_wt=0.0, maxiter=1000)
        except Exception:
            # As a last resort try GLM without regularization (may still fail, but we bubble up)
            model_fit = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Try to compute cluster-robust standard errors clustered by SiteID
    try:
        clustered_res = model_fit.get_robustcov_results(cov_type='cluster', groups=df['SiteID'])
    except Exception:
        clustered_res = model_fit

    return clustered_res