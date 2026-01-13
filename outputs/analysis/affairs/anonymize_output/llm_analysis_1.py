from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
from scipy.stats import norm

# Optional top-level read (wrapped to avoid import-time errors)
try:
    _input_path = '/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/anonymize_output/affairs.csv'
    _df_example = pd.read_csv(_input_path)
except Exception:
    _df_example = None


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the original DataFrame into the columns required for modeling.

    Produces the final dataframe with required columns:
    - AffairCount (float)
    - AnyAffair (int)
    - Children (int)
    - Male (int)
    - Age, YearsMarried, Religiosity, Education, Occupation, MaritalSatisfaction

    The function is robust to multiple possible input column names (e.g., 'feature2',
    'affairs', 'affair', etc.). It will pick the first matching source column found.
    Rows with missing values in required final columns are dropped.
    """
    df = df.copy()

    # Helper: map of lowercased existing columns to actual column names
    col_map = {c.lower(): c for c in df.columns}

    def find_col(candidates: List[str]) -> Optional[str]:
        """
        Return the first column name from df that matches any candidate (case-insensitive).
        If none found, return None.
        """
        for cand in candidates:
            if cand is None:
                continue
            lc = cand.lower()
            if lc in col_map:
                return col_map[lc]
        return None

    # Candidate source column names for each conceptual variable
    affair_src = find_col(['feature2', 'affairs', 'affair', 'affaircount', 'AffairCount'])
    children_src = find_col(['feature6', 'children', 'has_children', 'child', 'Child', 'Children'])
    gender_src = find_col(['feature3', 'gender', 'sex', 'male', 'Male'])
    age_src = find_col(['feature4', 'age', 'Age'])
    years_src = find_col(['feature5', 'yearsmarried', 'yrs_married', 'years_married', 'YearsMarried'])
    relig_src = find_col(['feature7', 'religious', 'religiosity', 'Religiosity'])
    educ_src = find_col(['feature8', 'education', 'Education'])
    occu_src = find_col(['feature9', 'occupation', 'Occupation'])
    rating_src = find_col(['feature10', 'rating', 'maritalsatisfaction', 'marital_satisfaction', 'MaritalSatisfaction'])

    # Build final columns. If source not found, create columns with NaN so dropna will handle.
    # AffairCount: numeric mapping of source values (use as-is)
    if affair_src is not None:
        df['AffairCount'] = pd.to_numeric(df[affair_src], errors='coerce')
    else:
        df['AffairCount'] = np.nan

    # AnyAffair: derived from AffairCount
    df['AnyAffair'] = (df['AffairCount'] > 0).astype('Int64')  # nullable int while building

    # Children: map yes/no or numeric 1/0
    if children_src is not None:
        col = df[children_src]
        # If values look like yes/no
        if col.dtype == object or col.dtype.name == 'category':
            mapped = col.astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0, 'y': 1, 'n': 0})
            # Also handle strings '0'/'1'
            mapped = mapped.fillna(col.astype(str).str.strip().replace({'0': 0, '1': 1}))
            df['Children'] = pd.to_numeric(mapped, errors='coerce').astype('Int64')
        else:
            # Numeric-ish
            df['Children'] = pd.to_numeric(col, errors='coerce').astype('Int64')
    else:
        df['Children'] = pd.NA

    # Male: map male/female or numeric coding
    if gender_src is not None:
        col = df[gender_src]
        if col.dtype == object or col.dtype.name == 'category':
            mapped = col.astype(str).str.strip().str.lower().map({'male': 1, 'm': 1, 'female': 0, 'f': 0})
            # Also handle strings '0'/'1'
            mapped = mapped.fillna(col.astype(str).str.strip().replace({'0': 0, '1': 1}))
            df['Male'] = pd.to_numeric(mapped, errors='coerce').astype('Int64')
        else:
            df['Male'] = pd.to_numeric(col, errors='coerce').astype('Int64')
    else:
        df['Male'] = pd.NA

    # Numeric controls: coerce to numeric, keep NA if not found
    df['Age'] = pd.to_numeric(df[age_src], errors='coerce') if age_src is not None else np.nan
    df['YearsMarried'] = pd.to_numeric(df[years_src], errors='coerce') if years_src is not None else np.nan
    df['Religiosity'] = pd.to_numeric(df[relig_src], errors='coerce') if relig_src is not None else np.nan
    df['Education'] = pd.to_numeric(df[educ_src], errors='coerce') if educ_src is not None else np.nan
    df['Occupation'] = pd.to_numeric(df[occu_src], errors='coerce') if occu_src is not None else np.nan
    df['MaritalSatisfaction'] = pd.to_numeric(df[rating_src], errors='coerce') if rating_src is not None else np.nan

    # Select required final columns and drop rows with missing values in those columns
    required_cols = ['AffairCount', 'Children', 'Age', 'YearsMarried', 'Male',
                     'Religiosity', 'Education', 'Occupation', 'MaritalSatisfaction']
    # Ensure columns exist in dataframe
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df.dropna(subset=required_cols)

    # Final keep columns, preserving AnyAffair as requested
    keep_cols = required_cols + ['AnyAffair']
    # Preserve original id if present (feature1 or id)
    id_col = find_col(['feature1', 'id', 'ID'])
    if id_col is not None:
        keep_cols = [id_col] + keep_cols

    # Ensure keep_cols exist in df (in case id_col was None)
    keep_cols = [c for c in keep_cols if c in df.columns]

    # Convert AnyAffair to int (non-null because dropna removed AffairCount NaNs)
    if 'AnyAffair' in df.columns:
        df['AnyAffair'] = df['AnyAffair'].astype(int)

    # Ensure final dataframe columns have expected dtypes
    # Children and Male as int
    df['Children'] = df['Children'].astype(int)
    df['Male'] = df['Male'].astype(int)

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Fit two models to assess the effect of having children on extramarital affairs:
      1) Tobit (censored regression) with censoring at 0 for AffairCount (primary).
      2) Logistic regression for AnyAffair (robustness / alternative specification).

    Returns a dictionary with fitted results objects: {'tobit': tobit_result, 'logit': logit_result}
    """
    results: Dict[str, Any] = {}

    # Prepare design matrix X for continuous outcome models (include intercept)
    covariates = ['Children', 'Age', 'YearsMarried', 'Male', 'Religiosity', 'Education', 'Occupation', 'MaritalSatisfaction']
    # Ensure the required covariates exist in df
    for cov in covariates:
        if cov not in df.columns:
            raise ValueError(f"Required covariate '{cov}' not found in dataframe passed to model().")

    X = sm.add_constant(df[covariates].astype(float), has_constant='add')
    y = df['AffairCount'].astype(float)

    # -- Tobit model (censoring at 0) --
    class TobitModel(GenericLikelihoodModel):
        def __init__(self, endog, exog, censoring=0.0, **kwargs):
            self.censoring = float(censoring)
            super(TobitModel, self).__init__(endog, exog, **kwargs)

        def nloglikeobs(self, params):
            # GenericLikelihoodModel expects negative log-likelihood per observation
            # params: [beta_0, beta_1, ..., beta_k, log_sigma]
            k = self.exog.shape[1]
            beta = params[:k]
            log_sigma = params[k]
            sigma = np.exp(log_sigma)

            mu = np.dot(self.exog, beta)
            y = np.asarray(self.endog).ravel()

            # For censored obs (y <= censoring) use log Phi( (c - mu)/sigma )
            # For uncensored obs (y > censoring) use log (1/sigma * phi((y-mu)/sigma))
            cens = (y <= self.censoring)
            uncens = ~cens

            ll = np.empty_like(y, dtype=float)
            # Uncensored contribution
            if np.any(uncens):
                z_uncens = (y[uncens] - mu[uncens]) / sigma
                ll[uncens] = -np.log(sigma) + norm.logpdf(z_uncens)
            # Censored contribution
            if np.any(cens):
                z_cens = (self.censoring - mu[cens]) / sigma
                ll[cens] = norm.logcdf(z_cens)

            # return negative log-likelihood per observation
            return -ll

        def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
            if start_params is None:
                # Initialize from OLS for beta and log(sigma)
                ols_res = sm.OLS(self.endog, self.exog).fit()
                beta0 = ols_res.params
                resid_std = np.nanstd(ols_res.resid)
                # Avoid log(0)
                if resid_std <= 0 or np.isnan(resid_std):
                    resid_std = 1.0
                sigma0 = np.log(resid_std)
                start_params = np.concatenate([beta0, [sigma0]])
            return super(TobitModel, self).fit(start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwds)

    try:
        tobit_mod = TobitModel(y.values, X.values, censoring=0.0)
        tobit_res = tobit_mod.fit(disp=False)
        results['tobit'] = tobit_res
    except Exception as e:
        # If Tobit fails to converge or raise an error, store the exception string
        results['tobit_error'] = str(e)

    # -- Logistic regression on AnyAffair (robustness) --
    X_bin = X  # same covariates
    if 'AnyAffair' not in df.columns:
        results['logit_error'] = "AnyAffair column not found in dataframe."
        return results

    y_bin = df['AnyAffair'].astype(int)

    try:
        logit_mod = sm.Logit(y_bin, X_bin)
        logit_res = logit_mod.fit(disp=False)
        results['logit'] = logit_res
    except Exception as e:
        results['logit_error'] = str(e)

    return results