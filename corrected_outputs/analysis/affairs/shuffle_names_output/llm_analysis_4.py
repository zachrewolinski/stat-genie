from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/affairs/shuffle_names_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and transform the raw Fair (affairs) dataset into a dataframe with the columns
    needed for modelling: AffairCount, HasChildren, Age, YearsMarried, Education,
    Religiousness, Occupation, MaritalSatisfaction.

    The function robustly identifies a column that indicates presence of children,
    coerces the affairs column to numeric (AffairCount), coerces controls to numeric,
    and drops rows missing critical values (AffairCount and HasChildren).

    Returns the dataframe with only the required columns for modelling.
    """
    df = df.copy()

    # 1) Create AffairCount from the 'affairs' column (coerce to numeric)
    df['AffairCount'] = pd.to_numeric(df.get('affairs'), errors='coerce')

    # Helper detection functions
    def _is_gender_like(series: pd.Series) -> bool:
        vals = series.dropna().astype(str).str.strip().str.lower().unique()
        return set(vals).issubset({'male', 'female', 'm', 'f'})

    def _is_yesno_like(series: pd.Series) -> bool:
        vals = series.dropna().astype(str).str.strip().str.lower().unique()
        allowed = {'yes', 'no', 'y', 'n', 'true', 'false', '1', '0'}
        return len(vals) > 0 and set(vals).issubset(allowed)

    def _is_binary_numeric(series: pd.Series) -> bool:
        try:
            numeric = pd.to_numeric(series.dropna(), errors='coerce')
            unique_numeric = set(np.unique(numeric.dropna()))
            return unique_numeric.issubset({0.0, 1.0})
        except Exception:
            return False

    def _is_small_count(series: pd.Series, max_count=10) -> bool:
        try:
            numeric = pd.to_numeric(series.dropna(), errors='coerce')
            if numeric.dropna().empty:
                return False
            # treat as small count if values are integers (nearly) and reasonably small
            rounded = np.round(numeric.dropna())
            is_integer = np.all(np.isclose(rounded, numeric.dropna()))
            return is_integer and numeric.dropna().min() >= 0 and numeric.dropna().max() <= max_count
        except Exception:
            return False

    # 2) Infer which column encodes whether there are children in the marriage.
    children_col = None

    # Prefer explicit 'children' column if it's plausibly a children indicator (not gender)
    if 'children' in df.columns:
        if not _is_gender_like(df['children']):
            children_col = 'children'

    # If not found, scan candidates 'children' then 'age' then any other columns heuristically
    if children_col is None:
        candidates = ['children', 'age']
        # include all columns as lower-priority candidates
        candidates += [col for col in df.columns if col not in candidates]
        seen = set()
        for cand in candidates:
            if cand in seen:
                continue
            seen.add(cand)
            if cand not in df.columns:
                continue
            series = df[cand].dropna()
            if series.empty:
                continue
            # skip obvious gender columns
            if _is_gender_like(series):
                continue
            # prefer yes/no-like strings
            if _is_yesno_like(series):
                children_col = cand
                break
            # prefer binary numeric 0/1
            if _is_binary_numeric(series):
                children_col = cand
                break
            # small integer counts (e.g., 0,1,2,3)
            if _is_small_count(series):
                children_col = cand
                break
            # As a last resort, if column named 'children' exists (even if odd), use it
            if cand == 'children':
                children_col = cand
                break

    # 3) Build HasChildren column robustly
    def map_to_has_children(series: pd.Series) -> pd.Series:
        s = series.copy()
        # strings / categorical
        if s.dtype == object or s.dtype.name == 'category':
            s_str = s.astype(str).str.strip().str.lower()
            # if values are male/female, cannot map -> return NaN
            if set(s_str.unique()).issubset({'male', 'female', 'm', 'f'}):
                return pd.Series([np.nan] * len(s), index=s.index)
            # map common yes/no tokens
            mapping = {
                'yes': 1, 'y': 1, 'true': 1, 't': 1, '1': 1,
                'no': 0, 'n': 0, 'false': 0, 'f': 0, '0': 0
            }
            s_mapped = s_str.replace(mapping)
            return pd.to_numeric(s_mapped, errors='coerce').astype(float)
        else:
            # numeric: decide whether it's binary indicator or count
            s_num = pd.to_numeric(s, errors='coerce')
            unique_numeric = set(np.unique(s_num.dropna())) if s_num.dropna().size > 0 else set()
            if unique_numeric.issubset({0.0, 1.0}):
                return s_num.astype(float)
            else:
                # otherwise treat >0 as having children
                return (s_num > 0).astype(float)

    if children_col is not None:
        df['HasChildren'] = map_to_has_children(df[children_col])
    else:
        # If we cannot find a children indicator, create HasChildren as NaN for all rows
        df['HasChildren'] = np.nan

    # 4) Controls: coerce relevant columns to numeric with sensible names
    # 'rating' in this dataset is the age coding (17.5,22,...)
    if 'rating' in df.columns:
        df['Age'] = pd.to_numeric(df['rating'], errors='coerce')
    else:
        df['Age'] = np.nan

    # Years married: prefer explicit 'yearsmarried' column
    if 'yearsmarried' in df.columns:
        df['YearsMarried'] = pd.to_numeric(df['yearsmarried'], errors='coerce')
    else:
        # some metadata versions store it in 'gender' column; attempt to coerce if plausible
        if 'gender' in df.columns:
            df['YearsMarried'] = pd.to_numeric(df['gender'], errors='coerce')
        else:
            df['YearsMarried'] = np.nan

    # Education
    if 'education' in df.columns:
        df['Education'] = pd.to_numeric(df['education'], errors='coerce')
    else:
        df['Education'] = np.nan

    # Religiousness
    if 'religiousness' in df.columns:
        df['Religiousness'] = pd.to_numeric(df['religiousness'], errors='coerce')
    else:
        df['Religiousness'] = np.nan

    # Occupation (numeric code)
    if 'occupation' in df.columns:
        df['Occupation'] = pd.to_numeric(df['occupation'], errors='coerce')
    else:
        df['Occupation'] = np.nan

    # Marital satisfaction is stored in 'rownames' in this schema
    if 'rownames' in df.columns:
        df['MaritalSatisfaction'] = pd.to_numeric(df['rownames'], errors='coerce')
    else:
        df['MaritalSatisfaction'] = np.nan

    # 5) Drop rows missing key model variables: AffairCount or HasChildren
    # If HasChildren could not be inferred and is all NaN, this will produce an empty dataframe.
    df = df.dropna(subset=['AffairCount', 'HasChildren'])

    # 6) Ensure numeric AffairCount and non-negative
    df['AffairCount'] = pd.to_numeric(df['AffairCount'], errors='coerce')
    df = df[df['AffairCount'].notna()]
    df = df[df['AffairCount'] >= 0]

    # Final dataframe keeps only the columns needed for modeling
    final_cols = [
        'AffairCount', 'HasChildren', 'Age', 'YearsMarried', 'Education',
        'Religiousness', 'Occupation', 'MaritalSatisfaction'
    ]
    # Ensure all final columns exist in the output (create with NaN if missing)
    for col in final_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a count regression to test the association between having children and the
    number of extramarital affairs. We use a Negative Binomial GLM to allow for
    overdispersion relative to Poisson.

    Model specification:
      AffairCount ~ HasChildren + Age + YearsMarried + Education + Religiousness + Occupation + MaritalSatisfaction

    Returns the fitted model results object (statsmodels GLMResults).
    """
    # ensure a copy so we don't modify input
    data = df.copy()

    # Drop rows with missing values in the dependent or key independent variable
    data = data.dropna(subset=['AffairCount', 'HasChildren'])

    # Prepare design matrix
    control_cols = ['Age', 'YearsMarried', 'Education', 'Religiousness', 'Occupation', 'MaritalSatisfaction']
    used_controls = [c for c in control_cols if c in data.columns]
    X = data[['HasChildren'] + used_controls].copy()

    # Coerce design matrix to numeric (controls may be NaN)
    X = X.apply(pd.to_numeric, errors='coerce')

    # Dependent variable
    y = pd.to_numeric(data['AffairCount'], errors='coerce')

    # Drop rows with any missing data in X or y (complete-case for the model)
    complete_mask = X.notna().all(axis=1) & y.notna()
    if complete_mask.sum() == 0:
        raise ValueError("No observations with complete data for AffairCount, HasChildren, and controls.")
    X = X.loc[complete_mask]
    y = y.loc[complete_mask]

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Fit Negative Binomial GLM (handles overdispersion better than Poisson)
    try:
        model_nb = sm.GLM(y, X, family=sm.families.NegativeBinomial())
        results = model_nb.fit()
    except Exception:
        # If NB fails for any reason (e.g., perfect separation or convergence), fall back to Poisson
        model_p = sm.GLM(y, X, family=sm.families.Poisson())
        results = model_p.fit()

    return results