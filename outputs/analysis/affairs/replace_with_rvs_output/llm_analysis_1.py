from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/replace_with_rvs_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Fair affairs dataset into a dataframe suitable for Tobit modeling.

    Output columns (kept/created):
      - affairs: numeric (dependent variable)
      - Children: binary (1 if children present, 0 if no)
      - Female: binary (1 if female, 0 if male)
      - Age: numeric
      - YearsMarried: numeric
      - Religiousness: numeric
      - Education: numeric
      - Occupation: numeric
      - Rating: numeric

    The function coerces types, maps categorical values to binaries, and drops rows with missing values in any of the columns used by the model.
    """
    df = df.copy()

    # Ensure dependent and key columns exist
    required_cols = ['affairs', 'children', 'gender', 'age', 'yearsmarried',
                     'religiousness', 'education', 'occupation', 'rating']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    # Standardize column name casing used in output
    # Rename source columns if necessary (keep original 'affairs')

    # Map children to binary: allow values 'yes'/'no', 'Yes'/'No', 1/0
    def map_children(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            xv = x.strip().lower()
            if xv in ['yes', 'y', 'true', '1']:
                return 1
            if xv in ['no', 'n', 'false', '0']:
                return 0
        try:
            # numeric
            if float(x) == 1:
                return 1
            if float(x) == 0:
                return 0
        except Exception:
            pass
        return np.nan

    df['Children'] = df['children'].apply(map_children)

    # Map gender to Female dummy (female=1, male=0). Handle common string forms.
    def map_female(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            xv = x.strip().lower()
            if xv in ['female', 'f', 'woman', 'women']:
                return 1
            if xv in ['male', 'm', 'man', 'men']:
                return 0
        try:
            # if numeric coding 1/0 observed
            if float(x) == 1:
                # ambiguous: assume 1 means female only if dataset documented so; but default to NaN
                return np.nan
        except Exception:
            pass
        return np.nan

    df['Female'] = df['gender'].apply(map_female)

    # Coerce numeric controls to numeric and copy to new standardized column names
    df['Age'] = pd.to_numeric(df['age'], errors='coerce')
    df['YearsMarried'] = pd.to_numeric(df['yearsmarried'], errors='coerce')
    df['Religiousness'] = pd.to_numeric(df['religiousness'], errors='coerce')
    df['Education'] = pd.to_numeric(df['education'], errors='coerce')
    df['Occupation'] = pd.to_numeric(df['occupation'], errors='coerce')
    df['Rating'] = pd.to_numeric(df['rating'], errors='coerce')

    # Dependent variable: ensure numeric
    df['affairs'] = pd.to_numeric(df['affairs'], errors='coerce')

    # Drop rows with missing values in any variables we will use
    model_cols = ['affairs', 'Children', 'Female', 'Age', 'YearsMarried',
                  'Religiousness', 'Education', 'Occupation', 'Rating']
    df = df.dropna(subset=model_cols).reset_index(drop=True)

    # Ensure types are integer where appropriate
    df['Children'] = df['Children'].astype(int)
    df['Female'] = df['Female'].astype(int)

    # (Optional) Keep only realistic affairs values: ensure non-negative
    df = df[df['affairs'] >= 0].copy()

    # Return trimmed dataframe for modeling
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a Tobit (left-censored at 0) model of affairs on Children and controls.

    Model specification:
      affairs_i* = X_i beta + u_i, u_i ~ N(0, sigma^2)
      observed affairs_i = max(0, affairs_i*)

    We include an interaction term between Children and Female (to examine whether the effect
    of having children differs by gender), and main controls.

    Returns the fitted results object (statsmodels GenericLikelihoodModelResults).
    """
    # Build design matrix
    exog_cols = [
        'Children',
        'Female',
        # interaction
        # we'll create a dedicated column for interaction
        'Children_x_Female',
        'Age',
        'YearsMarried',
        'Religiousness',
        'Education',
        'Occupation',
        'Rating'
    ]

    df = df.copy()
    df['Children_x_Female'] = df['Children'] * df['Female']

    # Add intercept inside the Tobit implementation (exog will include a constant)
    X = df[exog_cols]
    X = sm.add_constant(X, has_constant='add')
    y = df['affairs']

    # Generic Tobit implementation (left-censoring at 0)
    from statsmodels.base.model import GenericLikelihoodModel

    class TobitLeft(GenericLikelihoodModel):
        def __init__(self, endog, exog, left_censor=0.0, **kwds):
            super().__init__(endog, exog, **kwds)
            self.left_censor = left_censor

        def nloglikeobs(self, params):
            # return negative log-likelihood contributions for each observation
            # but GenericLikelihoodModel expects loglikeobs method; we implement nloglikeobs and loglike later
            return -self.loglikeobs(params)

        def loglikeobs(self, params):
            # params: [beta (k), log_sigma]
            k = self.exog.shape[1]
            beta = params[:k]
            log_sigma = params[k]
            sigma = np.exp(log_sigma)

            mu = np.dot(self.exog, beta)
            y = self.endog

            # For y > left_censor: density contribution
            z = (y - mu) / sigma
            # pdf and cdf values
            pdf_z = scipy.stats.norm.pdf(z)
            cdf_z_at_left = scipy.stats.norm.cdf((self.left_censor - mu) / sigma)

            # contribution arrays
            ll = np.zeros_like(y, dtype=float)

            mask_uncensored = y > self.left_censor
            # density contribution for uncensored obs: log(1/sigma * phi(z))
            ll[mask_uncensored] = -np.log(sigma) + np.log(pdf_z[mask_uncensored] + 1e-20)

            # censored observations: log( Phi((left - mu)/sigma) )
            mask_censored = ~mask_uncensored
            ll[mask_censored] = np.log(cdf_z_at_left[mask_censored] + 1e-20)

            return ll

        def loglike(self, params):
            return np.sum(self.loglikeobs(params))

    # Initial parameters: OLS for beta, log(sigma) from residuals
    ols_res = sm.OLS(y, X).fit()
    beta_init = ols_res.params.values
    resid = ols_res.resid
    sigma_init = max(1e-6, resid.std())
    start_params = np.concatenate([beta_init, [np.log(sigma_init)]])

    # Fit Tobit model
    mod = TobitLeft(y.values, X.values, left_censor=0.0)
    # use BFGS; set maxiter and disp
    res = mod.fit(start_params=start_params, method='bfgs', disp=False)

    # Attach pandas-friendly summary: build a results summary similar to statsmodels
    # The returned object is a GenericLikelihoodModelResults instance
    # Caller can inspect res.summary() or res.params
    return res

# Example usage (outside this function):
# df_trans = transform(raw_df)
# results = model(df_trans)
# print(results.summary())


