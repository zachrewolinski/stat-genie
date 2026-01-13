from typing import Any, Iterable, Optional
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm


def _find_source_column(df: pd.DataFrame, candidates: Iterable[str], extra_patterns: Optional[Iterable[str]] = None) -> Optional[str]:
    """
    Robustly find a plausible source column name in df given candidate names and optional extra regex patterns.

    Matching strategy (in order):
    1. Exact match to stripped column name.
    2. Case-insensitive match to stripped column name.
    3. Match against a simplified normalized form (lowercase, non-alnum -> '_').
    4. Regex search of original column names using extra_patterns.
    5. Regex search of normalized column names using extra_patterns.
    """
    cols = list(df.columns)

    # Build normalized mappings: several variants -> original column
    norm_map = {}
    for col in cols:
        col_str = str(col)
        stripped = col_str.strip()
        lowered = stripped.lower()
        simplified = re.sub(r'[^a-z0-9]+', '_', lowered).strip('_')
        # Keep the first seen mapping for each normalized key
        for key in (stripped, lowered, simplified):
            if key not in norm_map:
                norm_map[key] = col

    # 1. Exact match to stripped column name
    for c in candidates:
        c_str = str(c).strip()
        for col in cols:
            if str(col).strip() == c_str:
                return col

    # 2. Case-insensitive stripped match
    for c in candidates:
        c_lower = str(c).strip().lower()
        if c_lower in norm_map:
            return norm_map[c_lower]

    # 3. Simplified normalized candidate match
    for c in candidates:
        c_simpl = re.sub(r'[^a-z0-9]+', '_', str(c).strip().lower()).strip('_')
        if c_simpl in norm_map:
            return norm_map[c_simpl]

    # 4. Regex search through original column names (if provided)
    if extra_patterns:
        for pat in extra_patterns:
            try:
                regex = re.compile(pat, flags=re.I)
            except re.error:
                # skip invalid patterns
                continue
            for col in cols:
                if regex.search(str(col)):
                    return col

    # 5. Regex search through normalized keys
    if extra_patterns:
        for pat in extra_patterns:
            try:
                regex = re.compile(pat, flags=re.I)
            except re.error:
                continue
            for key, orig in norm_map.items():
                if regex.search(key):
                    return orig

    return None


def _normalize_gender(series: pd.Series) -> pd.Series:
    # Normalize common gender encodings to 'Girl' / 'Boy' or preserve other labels
    def map_gender(val):
        if pd.isna(val):
            return pd.NA
        v = str(val).strip().lower()
        if v in {'girl', 'female', 'f', 'g'}:
            return 'Girl'
        if v in {'boy', 'male', 'm', 'b'}:
            return 'Boy'
        # If it's already 'girl'/'boy' in any case
        if v in {'girl', 'boy'}:
            return v.capitalize()
        return str(val).strip()  # preserve other labels
    return series.map(map_gender).astype('category')


def _to_int_indicator(series: pd.Series) -> pd.Series:
    # Convert booleans, 'True'/'False', '1'/'0', numeric, etc. to nullable Int64 (0/1)
    if pd.api.types.is_bool_dtype(series):
        return series.astype('Int64')
    # Preserve NA values from the original series
    s = series.copy().astype(object)
    # Convert to string carefully, preserving NaN
    def to_str(x):
        if pd.isna(x):
            return 'nan'
        return str(x)
    s = s.map(to_str).str.strip().str.lower()
    def map_val(v):
        if v in {'nan', 'none', ''}:
            return pd.NA
        if v in {'1', '1.0', 'true', 't', 'yes', 'y'}:
            return 1
        if v in {'0', '0.0', 'false', 'f', 'no', 'n'}:
            return 0
        try:
            fv = float(v)
            return 1 if fv != 0 else 0
        except Exception:
            return pd.NA
    return s.map(map_val).astype('Int64')


def _auto_detect_choice_column(df: pd.DataFrame) -> Optional[str]:
    """
    Heuristic detection of a choice/response column when explicit name matching fails.
    Looks for columns with small number of unique values and values consistent with choice coding.
    """
    candidates = []
    n = len(df)
    for col in df.columns:
        series = df[col]
        # Skip columns that are obviously identifiers or high cardinality numeric
        unique_non_na = series.dropna().unique()
        nunique = len(unique_non_na)
        if nunique == 0:
            continue
        # Score based on whether values look like choices: numeric 1/2/3 or strings like 'majority'
        score = 0.0
        # Proportion of values that are integers 1-3
        cnt_good = 0
        cnt_total = 0
        for v in series.dropna().head(1000):  # sample up to first 1000 non-na for speed
            cnt_total += 1
            vs = str(v).strip().lower()
            if re.fullmatch(r'[123]', vs):
                cnt_good += 1
                continue
            if vs in {'majority', 'maj', 'major', 'minority', 'min', 'minor', 'undemonstrated', 'undem'}:
                cnt_good += 1
                continue
            if re.search(r'\b(majority|minority|undem)', vs):
                cnt_good += 1
                continue
            # numbers embedded in strings also count
            if re.search(r'\d', vs):
                cnt_good += 0.5
        if cnt_total > 0:
            prop = cnt_good / cnt_total
        else:
            prop = 0.0
        # Favor columns with small number of unique values (2-5)
        uniq_score = 1.0 if 2 <= nunique <= 5 else (0.5 if nunique <= 10 else 0.0)
        score = prop * 0.7 + uniq_score * 0.3
        candidates.append((score, col))
    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda x: x[0])
    best_score, best_col = candidates[0]
    # require a minimum confidence
    if best_score >= 0.35:
        return best_col
    return None


def _auto_detect_site_column(df: pd.DataFrame) -> Optional[str]:
    """
    Heuristic detection of a site/location column when explicit name matching fails.
    Prefer columns with moderate cardinality and textual values.
    """
    n = len(df)
    candidates = []
    for col in df.columns:
        series = df[col]
        nunique = series.nunique(dropna=True)
        if nunique == 0:
            continue
        # Skip columns that look like per-row unique identifiers
        if nunique > max(1, n * 0.9):
            continue
        # Score name matches
        name = str(col)
        name_score = 1.0 if re.search(r'site|location|lab|country|city', name, flags=re.I) else 0.0
        # Score based on type (textual/categorical better)
        type_score = 1.0 if (pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series)) else 0.5
        # Score based on cardinality being reasonable (2 <= nunique <= n/2)
        if 2 <= nunique <= max(2, n // 2):
            card_score = 1.0
        else:
            card_score = 0.5 if nunique <= max(2, n) else 0.0
        score = 0.5 * name_score + 0.3 * type_score + 0.2 * card_score
        candidates.append((score, col))
    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda x: x[0])
    best_score, best_col = candidates[0]
    if best_score >= 0.3:
        return best_col
    return None


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset to the analysis-ready dataframe.

    Produces (at minimum) the following final columns used in modeling:
    - ChoseMajority: binary dependent variable (1 if chose majority-demonstrated option, else 0)
    - Age_c: mean-centered age in years
    - Site: categorical site label used for fixed-effect site terms
    - Gender: categorical ('Girl' / 'Boy' or other category labels preserved)
    - MajorityFirst: indicator (0/1) whether majority was demonstrated first

    The function is robust to several possible raw column namings by
    checking common candidate names for each required input.
    """
    df = df.copy()

    # Candidate raw column names for each required conceptual input
    choice_candidates = ['Choice', 'choice', 'response', 'Response', 'feature1', 'feature_1', 'resp', 'answer', 'selection']
    age_candidates = ['Age', 'age', 'age_years', 'age_year', 'feature3', 'feature_3', 'yrs', 'years']
    siteid_candidates = ['SiteID', 'siteid', 'Site_Id', 'site_id', 'feature5', 'feature_5', 'site', 'location', 'lab', 'country']
    gender_candidates = ['Gender', 'gender', 'sex', 'Sex', 'feature2', 'feature_2']
    majorityfirst_candidates = ['MajorityFirst', 'majorityfirst', 'Majority_First', 'majority_first', 'feature4', 'feature_4', 'order_first', 'order', 'first']

    # Use regex patterns as a fallback to locate plausible columns
    choice_patterns = ['choice', 'response', 'resp', 'select', 'answer']
    age_patterns = ['age', 'years', 'yrs']
    site_patterns = ['site', 'location', 'lab', 'country', 'city']
    gender_patterns = ['gender', r'\bsex\b']
    majorityfirst_patterns = ['majority', 'order', 'first', 'presentation_order']

    src_choice = _find_source_column(df, choice_candidates, extra_patterns=choice_patterns)
    src_age = _find_source_column(df, age_candidates, extra_patterns=age_patterns)
    src_siteid = _find_source_column(df, siteid_candidates, extra_patterns=site_patterns)
    src_gender = _find_source_column(df, gender_candidates, extra_patterns=gender_patterns)
    src_majorityfirst = _find_source_column(df, majorityfirst_candidates, extra_patterns=majorityfirst_patterns)

    # If strict matching failed for some key columns, try heuristic detection
    if src_choice is None:
        src_choice = _auto_detect_choice_column(df)
    if src_siteid is None:
        src_siteid = _auto_detect_site_column(df)
    # For other fields, keep original behavior (no heuristics) but allow minor fallback
    if src_age is None:
        # try a simple heuristic: pick numeric column with plausible age range
        for col in df.columns:
            ser = pd.to_numeric(df[col], errors='coerce')
            if ser.notna().sum() == 0:
                continue
            med = ser.median(skipna=True)
            if pd.notna(med) and 0 < med < 100:
                # choose numeric column with median between 2 months and 100 years (practical)
                src_age = col
                break
    if src_gender is None:
        # try to find a low-cardinality text column
        for col in df.columns:
            ser = df[col]
            if ser.dropna().nunique() <= 10 and ser.dropna().nunique() > 1 and ser.dtype == object:
                # basic heuristic
                src_gender = col
                break
    if src_majorityfirst is None:
        # try to find binary-like column
        for col in df.columns:
            ser = df[col]
            # Check for boolean-like values
            vals = ser.dropna().unique()
            if len(vals) <= 3:
                lower_vals = {str(v).strip().lower() for v in vals}
                if lower_vals & {'first', 'true', '1', 'yes', 'y', 'majority'} or lower_vals & {'second', 'false', '0', 'no', 'n'}:
                    src_majorityfirst = col
                    break

    missing_sources = {}
    if src_choice is None:
        missing_sources['Choice'] = choice_candidates
    if src_age is None:
        missing_sources['Age'] = age_candidates
    if src_siteid is None:
        missing_sources['SiteID'] = siteid_candidates
    if src_gender is None:
        missing_sources['Gender'] = gender_candidates
    if src_majorityfirst is None:
        missing_sources['MajorityFirst'] = majorityfirst_candidates

    if missing_sources:
        missing_msg = ', '.join(f"{k} (tried: {v})" for k, v in missing_sources.items())
        raise ValueError(f"Could not find required input columns in the dataframe: {missing_msg}")

    # Create standardized intermediate columns from detected source columns
    df['Choice_raw'] = df[src_choice]
    df['Age_raw'] = df[src_age]
    df['SiteID_raw'] = df[src_siteid]  # keep raw to decide formatting
    df['Gender_raw'] = df[src_gender]
    df['MajorityFirst_raw'] = df[src_majorityfirst]

    # Coerce types and normalize fields

    # Age: numeric
    df['Age'] = pd.to_numeric(df['Age_raw'], errors='coerce')

    # Choice: attempt to map to integer codes where 2 == majority
    def map_choice(v):
        if pd.isna(v):
            return pd.NA
        # if already numeric-like
        try:
            fv = float(v)
            if np.isfinite(fv):
                return int(round(fv))
        except Exception:
            pass
        s = str(v).strip().lower()
        if s in {'majority', 'maj', 'major'}:
            return 2
        if s in {'minority', 'min', 'minor'}:
            return 3
        if s in {'undemonstrated', 'undem'}:
            return 1
        if s in {'1', '2', '3'}:
            try:
                return int(s)
            except Exception:
                return pd.NA
        # fallback: try to extract first integer from string
        m = re.search(r'(\d+)', s)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return pd.NA
        return pd.NA

    df['Choice'] = df['Choice_raw'].map(map_choice).astype('Int64')

    # MajorityFirst: indicator 0/1
    df['MajorityFirst'] = _to_int_indicator(df['MajorityFirst_raw'])

    # Gender normalized categorical
    df['Gender'] = _normalize_gender(df['Gender_raw'])

    # Construct Site label column:
    # Work from the raw value (don't coerce to str too early)
    def make_site_label_raw(v):
        if pd.isna(v):
            return pd.NA
        s = str(v).strip()
        if s == '' or s.lower() == 'none' or s.lower() == 'nan':
            return pd.NA
        if re.fullmatch(r'Site_\d+', s):
            return s
        if re.fullmatch(r'\d+', s):
            return 'Site_' + s
        return 'Site_' + re.sub(r'\s+', '_', s)
    df['Site'] = df['SiteID_raw'].map(make_site_label_raw).astype('category')

    # Drop rows with missing critical inputs after parsing
    df = df.dropna(subset=['Choice', 'Age', 'Site', 'Gender', 'MajorityFirst'])

    # Create dependent variable: ChoseMajority (1 if Choice == 2, else 0)
    # Use numpy arrays to ensure numpy dtypes (avoid pandas nullable dtypes for patsy)
    df['ChoseMajority'] = (df['Choice'].to_numpy() == 2).astype('int64')

    # Mean-center age
    df['Age_c'] = df['Age'] - df['Age'].mean()

    # Ensure MajorityFirst is a plain numpy integer dtype (not pandas nullable) for modeling compatibility
    if 'MajorityFirst' in df.columns:
        df['MajorityFirst'] = df['MajorityFirst'].astype('int64')

    # Ensure Gender and Site are categorical types (they already are), and final columns exist
    cols_needed = ['ChoseMajority', 'Age_c', 'Site', 'Gender', 'MajorityFirst']
    missing_final = [c for c in cols_needed if c not in df.columns]
    if missing_final:
        raise ValueError(f"Expected final column(s) not produced by transform: {missing_final}")

    # Return dataframe containing at least the final columns plus useful metadata
    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (binomial GLM) predicting the probability of choosing the majority option.

    Model formula:
      ChoseMajority ~ Age_c * C(Site) + C(Gender) + MajorityFirst

    Returns the fitted GLM results object.
    """
    import statsmodels.formula.api as smf

    required = ['ChoseMajority', 'Age_c', 'Site', 'Gender', 'MajorityFirst']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for modeling: {missing}")

    # Ensure the data types are compatible with patsy/statsmodels:
    # - ChoseMajority and MajorityFirst should be numeric numpy dtypes (int)
    # - Age_c should be float
    df = df.copy()
    df['ChoseMajority'] = pd.to_numeric(df['ChoseMajority'], errors='coerce').astype('int64')
    df['MajorityFirst'] = pd.to_numeric(df['MajorityFirst'], errors='coerce').astype('int64')
    df['Age_c'] = pd.to_numeric(df['Age_c'], errors='coerce').astype('float64')
    # Site and Gender should be categorical (pandas.Categorical); leave as-is if so
    if not pd.api.types.is_categorical_dtype(df['Site']):
        df['Site'] = df['Site'].astype('category')
    if not pd.api.types.is_categorical_dtype(df['Gender']):
        df['Gender'] = df['Gender'].astype('category')

    formula = 'ChoseMajority ~ Age_c * C(Site) + C(Gender) + MajorityFirst'
    model_fit = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
    results = model_fit.fit()

    return results