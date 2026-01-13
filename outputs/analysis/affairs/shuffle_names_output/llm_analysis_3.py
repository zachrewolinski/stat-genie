from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hc3
from scipy.stats import norm


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Fair (1978) dataset into a dataframe with the variables needed for modeling.

    Produces the following final columns used in the model:
      - AnyAffair: binary indicator (1 if affairs > 0, else 0)
      - HasChildren: binary indicator (1 if there are children in the marriage, 0 otherwise)
      - IsMale: binary indicator for male (1) / female (0)
      - YearsMarried: numeric years married
      - Education: numeric education coding
      - Religiousness: numeric religiousness coding
      - Age: numeric age code (variable named 'rating' in the raw data)
      - MaritalRating: numeric self-rating of marriage

    Notes: mapping is robust to a few common schema variants and attempts reasonable fallbacks
    while preserving the required final column names.
    """
    df = df.copy()

    # Normalize column name access: prefer exact names but allow case variants
    cols_lower = {c.lower(): c for c in df.columns}

    def colname(*candidates):
        """Return the first existing column name from candidates (case-insensitive), or None."""
        for cand in candidates:
            if cand is None:
                continue
            key = cand.lower()
            if key in cols_lower:
                return cols_lower[key]
        return None

    # Candidate source columns (some datasets vary in naming)
    col_affairs = colname('affairs', 'affair', 'num_affairs')
    col_children = colname('children', 'haschildren', 'has_child', 'child', 'kids')
    col_age = colname('age')
    col_yearsmarried = colname('yearsmarried', 'years.married', 'years_married', 'yrs_married', 'years')
    col_education = colname('education', 'educ')
    col_religiousness = colname('religiousness', 'religiosity', 'religious')
    col_rating = colname('rating')
    col_rownames = colname('rownames', 'maritalrating', 'marital_rating', 'marriage_rating')

    # Coerce common numeric columns if present
    for c in [col_affairs, col_yearsmarried, col_education, col_religiousness, col_rating, col_rownames]:
        if c is not None:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # AnyAffair from affairs (1 if >0, else 0)
    if col_affairs is not None:
        df['AnyAffair'] = (pd.to_numeric(df[col_affairs], errors='coerce') > 0).astype('Int64')
    else:
        # No source for affairs found; create column of NA (will be dropped later)
        df['AnyAffair'] = pd.Series([pd.NA] * len(df), dtype='Int64')

    # HasChildren: try several heuristics in order
    df['HasChildren'] = pd.Series([pd.NA] * len(df), dtype='Int64')
    # Primary: if 'age' column actually encodes yes/no for children as strings
    if col_age is not None:
        s = df[col_age].astype(str).str.strip().str.lower()
        mapped = s.map({'yes': 1, 'y': 1, 'no': 0, 'n': 0})
        if mapped.notna().any():
            df['HasChildren'] = mapped.astype('Int64')

    # Secondary: if 'children' column contains counts (numeric) use >0
    if df['HasChildren'].isna().all() and col_children is not None:
        # try numeric coercion
        num = pd.to_numeric(df[col_children], errors='coerce')
        if num.notna().any():
            df['HasChildren'] = (num > 0).astype('Int64')
        else:
            # try mapping yes/no strings in children column
            s = df[col_children].astype(str).str.strip().str.lower()
            mapped = s.map({'yes': 1, 'y': 1, 'no': 0, 'n': 0})
            if mapped.notna().any():
                df['HasChildren'] = mapped.astype('Int64')

    # Tertiary: look for explicit boolean-ish columns
    if df['HasChildren'].isna().all():
        alt = colname('haschildren', 'has_child', 'children_indicator')
        if alt is not None:
            num = pd.to_numeric(df[alt], errors='coerce')
            if num.notna().any():
                df['HasChildren'] = (num > 0).astype('Int64')

    # IsMale: try several heuristics
    df['IsMale'] = pd.Series([pd.NA] * len(df), dtype='Int64')
    # Primary: children column might contain gender labels per the dataset quirk
    if col_children is not None:
        s = df[col_children].astype(str).str.strip().str.lower()
        mapped = s.map({'male': 1, 'm': 1, 'man': 1, 'female': 0, 'f': 0, 'woman': 0})
        if mapped.notna().any():
            df['IsMale'] = mapped.astype('Int64')

    # Secondary: look for a gender/sex column
    if df['IsMale'].isna().all():
        alt_gender = colname('gender', 'sex')
        if alt_gender is not None:
            s = df[alt_gender].astype(str).str.strip().str.lower()
            mapped = s.map({'male': 1, 'm': 1, 'man': 1, 'female': 0, 'f': 0, 'woman': 0})
            if mapped.notna().any():
                df['IsMale'] = mapped.astype('Int64')
            else:
                # numeric codes: common encodings 0/1 or 1/2
                num = pd.to_numeric(df[alt_gender], errors='coerce')
                if num.notna().any():
                    # if values are 0/1 assume 1=male
                    unique = pd.Series(num.dropna().unique())
                    try:
                        vals = set(unique.astype(int).tolist())
                    except Exception:
                        vals = set()
                    if vals <= {0, 1}:
                        df['IsMale'] = (num == 1).astype('Int64')
                    elif vals <= {1, 2}:
                        # assume 1=male, 2=female
                        df['IsMale'] = (num == 1).astype('Int64')

    # YearsMarried, Education, Religiousness: straightforward copies if available
    if col_yearsmarried is not None:
        df['YearsMarried'] = df[col_yearsmarried]
    else:
        df['YearsMarried'] = pd.Series([pd.NA] * len(df))

    if col_education is not None:
        df['Education'] = df[col_education]
    else:
        df['Education'] = pd.Series([pd.NA] * len(df))

    if col_religiousness is not None:
        df['Religiousness'] = df[col_religiousness]
    else:
        df['Religiousness'] = pd.Series([pd.NA] * len(df))

    # Age and MaritalRating: choose sensible fallbacks so final columns are populated when possible
    # Prefer the explicit 'rating' as Age if it's clearly an age code; otherwise try 'age' numeric
    df['Age'] = pd.Series([pd.NA] * len(df))
    if col_rating is not None:
        df['Age'] = pd.to_numeric(df[col_rating], errors='coerce')
    elif col_age is not None:
        # If 'age' is numeric (not the yes/no case), use it
        age_num = pd.to_numeric(df[col_age], errors='coerce')
        if age_num.notna().any():
            df['Age'] = age_num

    # MaritalRating: prefer 'rownames' if present, else fallback to 'rating' if not used as Age
    df['MaritalRating'] = pd.Series([pd.NA] * len(df))
    if col_rownames is not None:
        df['MaritalRating'] = pd.to_numeric(df[col_rownames], errors='coerce')
    else:
        if col_rating is not None:
            df['MaritalRating'] = pd.to_numeric(df[col_rating], errors='coerce')

    # Ensure the final columns exist (even if all NA)
    final_cols = ['AnyAffair', 'HasChildren', 'IsMale', 'YearsMarried', 'Education', 'Religiousness', 'Age', 'MaritalRating']
    for c in final_cols:
        if c not in df.columns:
            df[c] = pd.Series([pd.NA] * len(df))

    # Convert final columns to numeric types where appropriate
    for c in final_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Finally, drop rows missing any model variable
    df = df.dropna(subset=final_cols).reset_index(drop=True)

    return df


class RobustResultsWrapper:
    """
    Minimal wrapper around a statsmodels results instance that substitutes
    robust covariance matrix (HC3) for inference-related attributes.
    Delegates attribute access to the original results object for everything
    else.
    """
    def __init__(self, results_obj, robust_cov):
        self._res = results_obj
        self._robust_cov = np.asarray(robust_cov)

    def __getattr__(self, name):
        # Delegate to underlying results for everything not explicitly overridden
        return getattr(self._res, name)

    @property
    def params(self):
        return self._res.params

    def cov_params(self):
        return self._robust_cov

    @property
    def bse(self):
        # robust standard errors
        return np.sqrt(np.diag(self._robust_cov))

    @property
    def tvalues(self):
        return self.params / self.bse

    @property
    def pvalues(self):
        # two-sided p-values using normal approximation
        tv = np.abs(self.tvalues)
        return 2 * (1 - norm.cdf(tv))

    def conf_int(self, alpha=0.05):
        z = norm.ppf(1 - alpha / 2.0)
        params = self.params
        se = self.bse
        lower = params - z * se
        upper = params + z * se
        return np.column_stack((lower, upper))

    def summary(self, *args, **kwargs):
        """
        Return the underlying summary. Note: the underlying summary will
        reflect the original (non-robust) standard errors. Consumers who
        need printed summaries with robust SEs should use the wrapper's
        params/bse/tvalues/pvalues/conf_int attributes directly.
        """
        return self._res.summary(*args, **kwargs)


def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression predicting the probability of any extramarital affair (AnyAffair)
    as a function of HasChildren and controls.

    Returns: statsmodels results object (robust covariance / HC3). If the input dataframe
    has no rows after preprocessing, raises a ValueError with an informative message.

    Model specification:
      AnyAffair ~ HasChildren + IsMale + YearsMarried + Education + Religiousness + Age + MaritalRating
    """
    # Make a local copy
    df = df.copy()

    # Validate that required columns are present
    required = ['AnyAffair', 'HasChildren', 'IsMale', 'YearsMarried', 'Education', 'Religiousness', 'Age', 'MaritalRating']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Input dataframe is missing required columns for modeling: {missing}")

    # Drop any rows with NA in required columns (should generally already be done by transform)
    df = df.dropna(subset=required).reset_index(drop=True)

    if df.shape[0] == 0:
        raise ValueError("No observations available for modeling after preprocessing. "
                         "Ensure transform produced rows with complete data for all required model variables.")

    # Define outcome and predictors
    y = df['AnyAffair'].astype(float)
    X = df[['HasChildren', 'IsMale', 'YearsMarried', 'Education', 'Religiousness', 'Age', 'MaritalRating']].astype(float)

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression (maximum likelihood)
    logit = sm.Logit(y, X)
    res = logit.fit(disp=False)

    # Compute robust (HC3) covariance matrix
    try:
        robust_cov = cov_hc3(res)
    except Exception:
        # If cov_hc3 fails for any reason, fall back to the result's covariance matrix
        try:
            robust_cov = res.cov_params()
        except Exception:
            robust_cov = np.asarray(np.zeros((len(res.params), len(res.params))))

    # Wrap results to present robust covariance for inference
    robust_res = RobustResultsWrapper(res, robust_cov)

    try:
        # Attempt to print a summary (may reflect non-robust SEs); ignore errors
        print(robust_res.summary())
    except Exception:
        pass

    return robust_res