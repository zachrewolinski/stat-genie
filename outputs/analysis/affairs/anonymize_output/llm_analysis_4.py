from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe with variables used in the models.

    Produces the REQUIRED final columns (exact names must be preserved):
    - AffairFreq: numeric outcome from survey coding (left-censored at 0)
    - HasChildren: binary indicator (1 = children in the marriage, 0 = no)
    - Female: binary gender (1 = female, 0 = male)
    - Age, Age_c
    - YearsMarried, YearsMarried_c
    - Religious, Education, Occupation, MaritalHappiness
    - AnyAffair: binary indicator (AffairFreq > 0)
    """

    df = df.copy()

    # Utility to find a source column in the raw dataframe given candidate names or substrings.
    def find_column(df, candidates):
        cols = list(df.columns)
        # direct match first
        for c in candidates:
            if c in cols:
                return c
        # case-insensitive exact match
        lower_map = {col.lower(): col for col in cols}
        for c in candidates:
            lc = c.lower()
            if lc in lower_map:
                return lower_map[lc]
        # substring match
        for c in candidates:
            lc = c.lower()
            for col in cols:
                if lc in col.lower():
                    return col
        return None

    # Specialized mappers for binary variables to handle different encodings robustly
    def map_has_children(series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            # common encodings: 0/1 or NaN; treat >0 as 1
            return series.map(lambda v: 1 if pd.notna(v) and float(v) > 0 else 0).astype('Int64')
        s = series.astype(str).str.strip().str.lower()
        pos = s.isin({'yes', 'y', '1', 'true', 't'}) | s.str.startswith('y')
        neg = s.isin({'no', 'n', '0', 'false', 'f'}) | s.str.startswith('n')
        out = pd.Series(index=series.index, dtype='Int64')
        out[pos] = 1
        out[neg] = 0
        # For anything else (e.g., numeric-looking strings), try numeric conversion
        if out.isna().any():
            try_num = pd.to_numeric(series, errors='coerce')
            out.loc[out.isna() & try_num.notna()] = try_num.loc[out.isna() & try_num.notna()].map(
                lambda v: 1 if v > 0 else 0
            )
        # If still NA, leave as NA (will be dropped later)
        return out.astype('Int64')

    def map_female(series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            # If values are 0/1 -> assume 1 is female
            unique = set(series.dropna().astype(int).unique())
            if unique <= {0, 1}:
                return series.map(lambda v: int(v)).astype('Int64')
            # If values appear like 1/2, assume 1 = female, 2 = male (common in some datasets)
            if unique <= {1, 2}:
                return series.map(lambda v: 1 if int(v) == 1 else 0).astype('Int64')
            # Fallback: treat >1 as missing
            return series.map(lambda v: 1 if v == 1 else (0 if v == 2 else pd.NA)).astype('Int64')

        s = series.astype(str).str.strip().str.lower()
        out = pd.Series(index=series.index, dtype='Int64')
        out[s.isin({'female', 'f', 'woman', 'w'})] = 1
        out[s.isin({'male', 'm', 'man'})] = 0
        # handle single-letter codes and obvious patterns
        out[s == '1'] = 1
        out[s == '0'] = 0
        # If still NA try numeric coercion
        if out.isna().any():
            try_num = pd.to_numeric(series, errors='coerce')
            out.loc[out.isna() & try_num.notna()] = try_num.loc[out.isna() & try_num.notna()].map(
                lambda v: 1 if v == 1 else (0 if v == 0 else pd.NA)
            )
        return out.astype('Int64')

    # Candidate names for each conceptual variable (these are only for locating raw columns;
    # final output columns must exactly match the required names and not be changed).
    col_candidates = {
        'AffairFreq': ['feature2', 'affairfreq', 'affair_freq', 'affair', 'affairs', 'extramarital_freq',
                       'extramarital', 'frequency', 'freq'],
        'HasChildren': ['feature6', 'haschildren', 'has_children', 'children', 'kids', 'has_kids'],
        'Female': ['feature3', 'female', 'sex', 'gender'],
        'Age': ['feature4', 'age', 'years_old'],
        'YearsMarried': ['feature5', 'yearsmarried', 'years_married', 'yrs_married', 'married_years'],
        'Religious': ['feature7', 'religious', 'religiosity', 'religiousness'],
        'Education': ['feature8', 'education', 'educ'],
        'Occupation': ['feature9', 'occupation', 'job', 'occ'],
        # include 'rating' as some datasets label marital happiness as 'rating'
        'MaritalHappiness': ['feature10', 'maritalhappiness', 'marital_happiness', 'happiness', 'marital', 'rating', 'marital_rating'],
    }

    # Locate each source column
    src_cols = {}
    for target, candidates in col_candidates.items():
        found = find_column(df, candidates)
        src_cols[target] = found

    # If some essential source columns are missing, raise a clear error
    missing = [t for t, c in src_cols.items() if c is None]
    if missing:
        raise KeyError(
            f"Required source columns not found for targets: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Map / convert each variable into the required final column names
    # AffairFreq: numeric, keep as is, treat non-numeric as NaN
    df['AffairFreq'] = pd.to_numeric(df[src_cols['AffairFreq']], errors='coerce')

    # HasChildren: binary 1/0
    df['HasChildren'] = map_has_children(df[src_cols['HasChildren']])

    # Female: binary 1=female, 0=male
    df['Female'] = map_female(df[src_cols['Female']])

    # Continuous controls
    df['Age'] = pd.to_numeric(df[src_cols['Age']], errors='coerce')
    df['YearsMarried'] = pd.to_numeric(df[src_cols['YearsMarried']], errors='coerce')
    df['Religious'] = pd.to_numeric(df[src_cols['Religious']], errors='coerce')
    df['Education'] = pd.to_numeric(df[src_cols['Education']], errors='coerce')
    df['Occupation'] = pd.to_numeric(df[src_cols['Occupation']], errors='coerce')
    df['MaritalHappiness'] = pd.to_numeric(df[src_cols['MaritalHappiness']], errors='coerce')

    # AnyAffair derived
    df['AnyAffair'] = (df['AffairFreq'] > 0).astype(int)

    # Required final columns list (must match the spec exactly)
    required_cols = [
        'AffairFreq', 'HasChildren', 'Female', 'Age', 'YearsMarried',
        'Religious', 'Education', 'Occupation', 'MaritalHappiness'
    ]

    # Drop rows with missing values in required modeling variables
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    # Ensure binary columns are concrete ints (after dropping NA rows)
    df['HasChildren'] = df['HasChildren'].astype(int)
    df['Female'] = df['Female'].astype(int)

    # Center continuous covariates
    df['Age_c'] = df['Age'] - df['Age'].mean()
    df['YearsMarried_c'] = df['YearsMarried'] - df['YearsMarried'].mean()

    # Keep only the final columns required (plus the centered versions)
    keep_cols = [
        'AffairFreq', 'HasChildren', 'Female', 'Age', 'Age_c', 'YearsMarried', 'YearsMarried_c',
        'Religious', 'Education', 'Occupation', 'MaritalHappiness', 'AnyAffair'
    ]

    return df[keep_cols].copy()


def model(df: pd.DataFrame) -> Any:
    """
    Fit a censored (Tobit) regression of AffairFreq on HasChildren and controls.
    Also fit a logistic regression of AnyAffair on the same predictors as a robustness check.

    Expects df to be the FINAL dataframe produced by transform().
    Returns a dictionary with the fitted Tobit and Logit result objects.
    """
    from statsmodels.base.model import GenericLikelihoodModel

    # Prepare design matrix
    predictors = ['HasChildren', 'Female', 'Age_c', 'YearsMarried_c',
                  'Religious', 'Education', 'Occupation', 'MaritalHappiness']

    X = df[predictors].astype(float)
    X = sm.add_constant(X)
    y = df['AffairFreq'].astype(float)

    # Tobit (left-censored at 0) implemented via GenericLikelihoodModel
    class Tobit(GenericLikelihoodModel):
        def __init__(self, endog, exog, left_censor=0.0, **kwargs):
            self.left_censor = left_censor
            super(Tobit, self).__init__(endog, exog, **kwargs)

        def loglike(self, params):
            # params: [beta_0, ..., beta_k, log_sigma]
            k = self.exog.shape[1]
            beta = params[:k]
            log_sigma = params[k]
            sigma = np.exp(log_sigma)

            Xb = np.dot(self.exog, beta)
            y = self.endog

            # For censored observations y <= left_censor
            cens = (y <= self.left_censor)
            uncens = ~cens

            # Avoid numerical issues by bounding cdf inputs
            z_cens = (self.left_censor - Xb[cens]) / (sigma + 1e-20)
            ll_cens = np.log(stats.norm.cdf(z_cens) + 1e-20)

            # For uncensored observations: density of normal
            z_uncens = (y[uncens] - Xb[uncens]) / (sigma + 1e-20)
            ll_uncens = -np.log(sigma + 1e-20) + stats.norm.logpdf(z_uncens)

            total_ll = np.sum(ll_cens) + np.sum(ll_uncens)
            return total_ll

        def start_params(self):
            # OLS start for betas
            ols_res = sm.OLS(self.endog, self.exog).fit()
            beta_start = np.asarray(ols_res.params, dtype=float)
            resid = ols_res.resid
            # ensure positive std and finite
            sigma_est = float(np.std(resid)) if np.isfinite(np.std(resid)) and np.std(resid) > 0 else 1.0
            sigma_start = np.log(sigma_est + 1e-8)
            return np.concatenate([beta_start, np.array([sigma_start], dtype=float)])

    # Fit Tobit
    tobit_model = Tobit(y.values, X.values, left_censor=0.0)
    # Ensure numeric start_params are provided to avoid a statsmodels issue where
    # the method reference could be used instead of actual start values.
    start = tobit_model.start_params()
    try:
        tobit_res = tobit_model.fit(start_params=start, method='bfgs', disp=False)
    except Exception:
        # fallback to default method if bfgs fails; still pass numeric start
        tobit_res = tobit_model.fit(start_params=start, disp=False)

    # Robustness: logistic regression on binary any-affair outcome
    y_bin = df['AnyAffair'].astype(int)
    logit_model = sm.Logit(y_bin, X)
    try:
        logit_res = logit_model.fit(disp=False)
    except Exception:
        # If perfect separation or convergence issues occur, use method='lbfgs'
        logit_res = logit_model.fit(method='lbfgs', disp=False)

    return {
        'tobit_results': tobit_res,
        'logit_results': logit_res,
        'predictors': predictors
    }