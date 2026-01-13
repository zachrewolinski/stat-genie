from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Attempt to read a CSV if running as a script; keep for compatibility but won't execute on import.
try:
    df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/anonymize_output/boxes.csv')
except Exception:
    df = None


def _find_column(df: pd.DataFrame, candidates):
    """
    Helper: find the first column in df whose lowercase name matches any candidate (also lowered).
    Returns the actual column name if found, otherwise None.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand is None:
            continue
        cl = cand.lower()
        if cl in cols_lower:
            return cols_lower[cl]
    # Also try substring match: candidate appears within column name
    for cand in candidates:
        cl = cand.lower()
        for col in df.columns:
            if cl in col.lower():
                return col
    return None


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset to a modeling-ready dataframe.

    Required final columns that this function guarantees (and that the model expects):
      - ChoiceMajority : int (1 if original outcome indicates majority chosen else 0)
      - Age : original age as float
      - Age_c : mean-centered age
      - Age_sq : squared mean-centered age
      - SiteID : categorical site identifier (string)
      - Gender : binary (1 = girl, 0 = boy)
      - MajorityFirst : int copy of demonstration-order indicator (0/1)

    This function is robust to several common alternative column namings in the raw CSV.
    It will search for plausible alternative names for the required raw features and copy them
    to internal columns named 'feature1'..'feature5' before performing transformations.
    """
    df = df.copy()

    # Define canonical raw feature names we expect to find (we will create these as copies if needed)
    canonical = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']

    # Candidate alternative names for each canonical feature (common variants)
    alternatives = {
        'feature1': ['feature1', 'feature_1', 'feature 1', 'outcome', 'choice', 'response', 'resp', 'result', 'feature.1', 'f1'],
        'feature2': ['feature2', 'feature_2', 'feature 2', 'gender', 'sex', 'sex_assigned', 'participant_sex', 'feature.2', 'f2'],
        'feature3': ['feature3', 'feature_3', 'feature 3', 'age', 'age_years', 'age_in_years', 'child_age', 'feature.3', 'f3'],
        'feature4': ['feature4', 'feature_4', 'feature 4', 'majority_first', 'demo_order', 'first_demo', 'order', 'presentation_order', 'feature.4', 'f4'],
        'feature5': ['feature5', 'feature_5', 'feature 5', 'site', 'siteid', 'site_id', 'siteID', 'location', 'study_site', 'feature.5', 'f5'],
    }

    # For each canonical feature, find a matching column in df and create a canonical-named copy
    for feat in canonical:
        col_found = _find_column(df, alternatives[feat])
        if col_found is None:
            # If not found, create the column with NaNs so downstream dropna will remove invalid rows.
            df[feat] = np.nan
        else:
            # Copy to canonical name (creates/overwrites)
            df[feat] = df[col_found]

    # Now drop rows missing any of the raw canonical features (these are required to construct final vars)
    required_raw = canonical
    df = df.dropna(subset=required_raw)

    # Dependent variable: chose the majority (feature1 == 2)
    # Some datasets may encode majority choice differently; here we follow the documented mapping:
    # original: 1=unchosen, 2=majority, 3=minority
    df['ChoiceMajority'] = (pd.to_numeric(df['feature1'], errors='coerce') == 2).astype(int)

    # Age and transformations
    # Accept numeric strings as well.
    df['Age'] = pd.to_numeric(df['feature3'], errors='coerce').astype(float)
    # mean-center age for interpretability
    # If Age column ended up all-NaN, mean() will be nan and results will be NaN; such rows will be dropped below.
    age_mean = df['Age'].mean()
    df['Age_c'] = df['Age'] - age_mean
    # quadratic term to allow non-linear development
    df['Age_sq'] = df['Age_c'] ** 2

    # Site / cultural context as categorical
    # Construct robust string labels for SiteID from feature5
    site_raw = df['feature5']

    def _to_site_label(x):
        if pd.isna(x):
            return np.nan
        # Try integer-like numeric
        try:
            num = float(x)
            if np.isfinite(num):
                # If it's integer-valued, prefer integer string
                if num.is_integer():
                    return str(int(num))
                return str(num)
        except Exception:
            pass
        s = str(x).strip()
        if s == '':
            return np.nan
        return s

    site_labels = [_to_site_label(x) for x in site_raw.tolist()]
    # assign and make categorical; NaNs will be removed by subsequent dropna on final columns
    df['SiteID'] = pd.Categorical(site_labels)

    # Gender: map to binary (1 = girl, 0 = boy)
    # Support multiple encodings: numeric (1=girl,2=boy), textual ('girl'/'boy', 'female'/'male')
    gender_raw = df['feature2']

    gender_mapped = pd.Series(index=df.index, dtype='float64')

    # Numeric mapping first
    gender_num = pd.to_numeric(gender_raw, errors='coerce')
    # If numeric mapping yields 1 or 2, map accordingly
    mask_num = gender_num.isin([1, 2])
    gender_mapped.loc[mask_num] = gender_num.loc[mask_num].map({1: 1, 2: 0})

    # Textual mapping for the rest
    mask_text = gender_mapped.isna()
    if mask_text.any():
        text_vals = gender_raw.astype(str).str.strip().str.lower()
        map_text = {
            'girl': 1, 'female': 1, 'f': 1, 'woman': 1,
            'boy': 0, 'male': 0, 'm': 0, 'man': 0
        }
        gender_mapped.loc[mask_text] = text_vals.map(map_text)

    # Final cast to integer type where possible
    df['Gender'] = gender_mapped.astype('Int64')  # nullable integer to preserve potential NAs

    # MajorityFirst: ensure integer 0/1
    maj_raw = df['feature4']

    # Try numeric coercion first
    maj_num = pd.to_numeric(maj_raw, errors='coerce')
    maj_mapped = maj_num.copy()

    # For non-numeric, map common textual variants
    mask_nonnum = maj_mapped.isna()
    if mask_nonnum.any():
        text_vals = maj_raw.astype(str).str.strip().str.lower()
        map_text = {
            '1': 1, '0': 0, 'yes': 1, 'no': 0, 'true': 1, 'false': 0,
            't': 1, 'f': 0, 'y': 1, 'n': 0, 'majority_first': 1, 'first': 1
        }
        maj_mapped.loc[mask_nonnum] = text_vals.map(map_text)

    # If values are booleans, convert True/False to 1/0
    if maj_mapped.isin([True, False]).any():
        maj_mapped = maj_mapped.replace({True: 1, False: 0})

    df['MajorityFirst'] = maj_mapped.astype('Int64')

    # Final check: drop any rows that may have become NA after transformations in the final columns
    final_cols = ['ChoiceMajority', 'Age', 'Age_c', 'Age_sq', 'SiteID', 'Gender', 'MajorityFirst']
    df = df.dropna(subset=final_cols)

    # Ensure final column types are as expected (concrete ints where possible)
    df['ChoiceMajority'] = df['ChoiceMajority'].astype(int)
    df['Age'] = df['Age'].astype(float)
    df['Age_c'] = df['Age_c'].astype(float)
    df['Age_sq'] = df['Age_sq'].astype(float)
    # SiteID is categorical already (pd.Categorical)
    # Convert categorical to have only observed categories (drop unused)
    if isinstance(df['SiteID'].dtype, pd.CategoricalDtype):
        df['SiteID'] = df['SiteID'].cat.remove_unused_categories()
    df['Gender'] = df['Gender'].astype(int)
    df['MajorityFirst'] = df['MajorityFirst'].astype(int)

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial (logistic) generalized linear model to predict the probability
    of choosing the majority option. The formula includes:
      - main effect of mean-centered age (Age_c) and Age_sq for nonlinearity
      - main effect of SiteID (C(SiteID)) to capture cultural differences
      - interaction Age_c:C(SiteID) to allow age slopes to differ by culture
      - controls: Gender and MajorityFirst

    Returns a dictionary with the fitted model and a clustered-robust-covariance
    version (clustered by SiteID) when available.
    """
    # Ensure required columns exist
    required = ['ChoiceMajority', 'Age_c', 'Age_sq', 'SiteID', 'Gender', 'MajorityFirst']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # If no rows, return empty results instead of attempting to fit
    if df.shape[0] == 0:
        return {'glm_result': None, 'glm_result_clustered_se': None}

    # If SiteID has no observed levels, avoid fitting (patsy will error); return empty results
    try:
        n_sites = df['SiteID'].nunique(dropna=True)
    except Exception:
        n_sites = 0
    if n_sites == 0:
        return {'glm_result': None, 'glm_result_clustered_se': None}

    # Formula: Age (linear + quadratic), site main effects, Age x Site interactions, plus controls
    formula = 'ChoiceMajority ~ Age_c + Age_sq + C(SiteID) + Age_c:C(SiteID) + Gender + MajorityFirst'

    # Fit GLM with binomial family (logistic regression)
    glm_binom = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
    res = glm_binom.fit()

    # Attempt to get clustered (by SiteID) robust standard errors
    try:
        res_cluster = res.get_robustcov_results(cov_type='cluster', groups=df['SiteID'])
    except Exception:
        res_cluster = None

    return {
        'glm_result': res,
        'glm_result_clustered_se': res_cluster
    }