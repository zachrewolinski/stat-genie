from typing import Any, Dict, FrozenSet, List, Literal, Optional, Set, Tuple
import re

import numpy as np
import pandas as pd
import sklearn  # noqa: F401
import scipy  # noqa: F401
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt  # noqa: F401
import pickle  # noqa: F401


def _normalize_name(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


def _find_column(df: pd.DataFrame, candidates: List[str], exclude: Optional[Set[str]] = None) -> Optional[str]:
    """
    Return the first column name from df whose normalized form matches or contains any
    of the normalized candidate strings. If none found, return None.

    The optional exclude set contains column names to skip (useful to avoid mapping the same
    source column to multiple targets).
    """
    if exclude is None:
        exclude = set()
    norm_cols = {col: _normalize_name(col) for col in df.columns if col not in exclude}
    for cand in candidates:
        nc = _normalize_name(cand)
        # exact match
        for col, ncol in norm_cols.items():
            if ncol == nc:
                return col
        # contains match
        for col, ncol in norm_cols.items():
            if nc in ncol or ncol in nc:
                return col
    # fallback: try to match by digits (e.g., feature1, feat1)
    for cand in candidates:
        m = re.search(r'\d+', str(cand))
        if m:
            digit = m.group(0)
            for col, ncol in norm_cols.items():
                if digit in ncol:
                    return col
    return None


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the dataframe used for modeling.
    Expected original columns may vary; this function will try to locate
    appropriate source columns and produce the final dataframe with columns:
      ['MajorityChoice', 'Age', 'Age_c', 'Female', 'MajorityFirst', 'Site', 'Choice', 'Gender']
    """
    df = df.copy()

    # Target intermediate/raw column names we need to have before computing derived vars
    expected_raw = ['Choice', 'Gender', 'Age', 'MajorityFirst', 'Site']

    # Determine which expected raw columns are missing and attempt to locate them
    missing_targets = [t for t in expected_raw if t not in df.columns]

    if missing_targets:
        mapping_candidates = {
            'Choice': [
                'feature1', 'choice', 'response', 'answer', 'feat1', 'y', 'outcome', 'result',
                'selection', 'selected', 'selected_option', 'option_selected', 'pick', 'option',
                'response_y', 'choice_y', 'yes_no', 'yesno', 'binary_choice'
            ],
            'Gender': [
                'feature2', 'gender', 'sex', 'sex_at_birth', 'feat2', 'participant_sex', 'male_female', 'm/f'
            ],
            'Age': [
                'feature3', 'age', 'age_years', 'ageyrs', 'years', 'feat3', 'child_age', 'age_in_months'
            ],
            'MajorityFirst': [
                'feature4', 'majorityfirst', 'majority_first', 'majority-first',
                'majorityfirstflag', 'order_majority', 'feat4', 'majority_first_shown', 'first_was_majority'
            ],
            'Site': [
                'feature5', 'site', 'site_id', 'location', 'lab', 'study_site', 'feat5', 'culture', 'country', 'sitecode'
            ]
        }

        rename_map: Dict[str, str] = {}
        used_found_cols: Set[str] = set()
        # Greedily assign distinct source columns to each missing target.
        for target in missing_targets:
            candidates = mapping_candidates.get(target, [])
            found = _find_column(df, candidates, exclude=used_found_cols)
            if found is None:
                # If no non-used column found, attempt one last time allowing reuse (helps when dataset is tiny)
                found = _find_column(df, candidates, exclude=set())
                if found is None:
                    raise KeyError(
                        f"Could not find a column for '{target}'. "
                        f"Looked for candidates: {candidates}. Available columns: {list(df.columns)}"
                    )
                # If found but already used for another target, treat as ambiguous and error to avoid silent reuse
                if found in used_found_cols:
                    raise KeyError(
                        f"Ambiguous mapping: column '{found}' would be used for multiple targets including '{target}'. "
                        f"Columns available: {list(df.columns)}"
                    )
            rename_map[found] = target
            used_found_cols.add(found)

        # Rename the located columns to the expected intermediate names
        if rename_map:
            df = df.rename(columns=rename_map)

    # Ensure that after mapping we have all required raw columns
    still_missing = [t for t in expected_raw if t not in df.columns]
    if still_missing:
        raise KeyError(f"Missing required raw columns after mapping: {still_missing}. Available: {list(df.columns)}")

    # Drop rows with missing essential values
    df = df.dropna(subset=expected_raw)

    # Dependent variable: did child choose the majority-demonstrated option?
    choice_series = df['Choice']
    if pd.api.types.is_numeric_dtype(choice_series):
        # Convert to numeric (preserve NaNs) and then interpret
        choice_numeric = pd.to_numeric(choice_series, errors='coerce')
        uniq = [v for v in pd.unique(choice_numeric) if pd.notna(v)]
        if 2 in uniq:
            df['MajorityChoice'] = (choice_numeric == 2).astype(int)
        else:
            unique_vals = sorted(uniq)
            if len(unique_vals) == 2:
                # map larger value to 1
                max_val = max(unique_vals)
                df['MajorityChoice'] = (choice_numeric == max_val).astype(int)
            else:
                # fallback: treat values > mean as majority (conservative)
                mean_val = choice_numeric.mean()
                df['MajorityChoice'] = (choice_numeric > mean_val).astype(int)
    else:
        # object dtype: check for textual markers
        lower_vals = choice_series.astype(str).str.lower()
        # common markers for majority/minority
        maj_mask = lower_vals.str.contains('major|maj|majority', na=False)
        df['MajorityChoice'] = maj_mask.astype(int)
        # handle common binary markers like 'y'/'n', 'yes'/'no', '1'/'0'
        if df['MajorityChoice'].sum() == 0:
            # if values look like yes/no or y/n map yes/y/1 to 1
            yes_mask = lower_vals.isin(['y', 'yes', '1', 'true', 't'])
            no_mask = lower_vals.isin(['n', 'no', '0', 'false', 'f'])
            if yes_mask.any() and no_mask.any():
                df['MajorityChoice'] = yes_mask.astype(int)
            elif lower_vals.nunique(dropna=True) == 2:
                # fallback: map the most frequent to 1 (conservative approach)
                most_freq = lower_vals.value_counts(dropna=True).idxmax()
                df['MajorityChoice'] = (lower_vals == most_freq).astype(int)
            else:
                # final fallback: keep zeros (no evidence of majority marker)
                df['MajorityChoice'] = (pd.Series(0, index=df.index)).astype(int)

    # Control: Female indicator (1 if girl, 0 if boy)
    gender_series = df['Gender']
    if pd.api.types.is_numeric_dtype(gender_series):
        # Original code expected 1 = girl, 2 = boy
        df['Female'] = (pd.to_numeric(gender_series, errors='coerce') == 1).astype(int)
    else:
        # object/string mapping
        s = gender_series.astype(str).str.lower()
        # Match full words first, then single-letter tokens to avoid matching 'male' in 'female' twice.
        df['Female'] = s.map(lambda x: 1 if any(tok in x for tok in ['female', 'girl', 'woman']) or x in ['f'] else 0).astype(int)

    # Ensure MajorityFirst is binary integer
    mf = df['MajorityFirst']
    if pd.api.types.is_numeric_dtype(mf):
        # Map non-zero to 1
        df['MajorityFirst'] = (pd.to_numeric(mf, errors='coerce').astype(float) != 0).astype(int)
    else:
        s = mf.astype(str).str.lower()
        df['MajorityFirst'] = s.map(lambda x: 1 if any(tok in x for tok in ['1', 'true', 'yes', 'y', 'majority', 'first']) else 0).astype(int)

    # Site as categorical variable (keep original ids but convert dtype)
    df['Site'] = df['Site'].astype('category')

    # Ensure Age is numeric
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    # Drop any rows with NaN age introduced by coercion
    df = df.dropna(subset=['Age'])

    # Center age for interpretability (used in interactions)
    df['Age_c'] = df['Age'] - df['Age'].mean()

    # Keep columns necessary for modeling (plus some originals for traceability)
    final_cols = ['MajorityChoice', 'Age', 'Age_c', 'Female', 'MajorityFirst', 'Site', 'Choice', 'Gender']
    # If some trace columns are missing (shouldn't be), create them as copies where possible
    for col in ['Choice', 'Gender']:
        if col not in df.columns:
            df[col] = np.nan

    return df[final_cols]


def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic (binomial) regression to model the probability of choosing the majority option.

    Model specification:
      MajorityChoice ~ Age_c * C(Site) + Female + MajorityFirst

    Returns results with cluster-robust standard errors clustered by Site where possible.
    """
    import statsmodels.api as sm  # local import to ensure availability in environments
    import statsmodels.formula.api as smf

    # Work on a copy
    df = df.copy()
    # Ensure required columns exist
    required = ['MajorityChoice', 'Age_c', 'Site', 'Female', 'MajorityFirst']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for modeling: {missing}")

    # Ensure Site is categorical for the formula interface
    df['Site'] = df['Site'].astype('category')

    formula = 'MajorityChoice ~ Age_c * C(Site) + Female + MajorityFirst'

    # Fit binomial GLM
    glm_mod = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
    glm_res = glm_mod.fit()

    # Obtain cluster-robust standard errors clustered by site id
    # If clustering fails for any reason, fall back to the original fit
    try:
        results = glm_res.get_robustcov_results(cov_type='cluster', groups=df['Site'])
    except Exception:
        results = glm_res

    # Print summary for quick inspection
    print(results.summary())

    return results