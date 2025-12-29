from typing import Any, Iterable, List, Optional
import re

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def _normalize(name: str) -> str:
    """Normalize column/candidate names for robust matching."""
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """
    Return the first column name in df whose normalized form matches any of the
    normalized candidates. Matching is done on exact normalized equality.
    """
    norm_to_col = { _normalize(col): col for col in df.columns }
    for cand in candidates:
        norm = _normalize(cand)
        if norm in norm_to_col:
            return norm_to_col[norm]
    return None


def _find_column_by_substring(df: pd.DataFrame, substrings: Iterable[str]) -> Optional[str]:
    """
    Return the first column name whose normalized form contains any of the given substrings.
    Used as a fallback when exact candidate names don't match.
    """
    norm_to_col = { _normalize(col): col for col in df.columns }
    # Normalize substrings for robust matching
    normalized_subs = [_normalize(s) for s in substrings]
    for key, col in norm_to_col.items():
        for sub in normalized_subs:
            if sub and sub in key:
                return col
    return None


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a modeling-ready dataframe. The final dataframe contains the columns:
      - MajorityChoice (DV: 1 if child chose majority option, 0 otherwise)
      - Age (raw age in years)
      - Age_c (mean-centered age)
      - Site (categorical site identifier)
      - IsBoy (control: 1 if boy, 0 if girl)
      - MajorityFirst (control: 0/1 whether majority was demonstrated first)

    The function is robust to inputs that either already use the final column names
    or use several common alternative/raw feature names. It will attempt to infer
    the correct source columns using a set of candidate names and simple heuristics.
    """
    df = df.copy()

    # Candidate sets for each conceptual variable (ordered by preference).
    age_candidates = ['Age', 'feature3', 'age_years', 'ageyears', 'years', 'childage']
    majorityfirst_candidates = ['MajorityFirst', 'feature4', 'majority_first', 'demonstrationorder', 'firstdemo']
    site_candidates = ['Site', 'feature5', 'site', 'location', 'country', 'culturalcontext']
    gender_candidates = ['Gender', 'feature2', 'gender', 'sex']
    choice_candidates = ['Choice', 'feature1', 'choice', 'selection', 'response']
    isboy_candidates = ['IsBoy', 'is_boy', 'isboy', 'boy']

    # Helper to locate column name in dataframe
    def locate(candidates: List[str], substr_fallback: List[str] = None) -> Optional[str]:
        # 1) Exact normalized match
        col = _find_column(df, candidates)
        if col is not None:
            return col
        # 2) Normalized substring fallback
        if substr_fallback:
            col = _find_column_by_substring(df, substr_fallback)
            if col is not None:
                return col
        # 3) Raw case-insensitive substring match on original column names
        lower_cols = [c.lower() for c in df.columns]
        cand_checks = [c.lower() for c in (list(candidates) + (substr_fallback or []))]
        for i, colname in enumerate(df.columns):
            col_lower = lower_cols[i]
            for cand in cand_checks:
                if not cand:
                    continue
                if cand in col_lower or col_lower in cand:
                    return colname
        # 4) As a last resort, if there's any column whose normalized name is a superset of any candidate norm
        cand_norms = [_normalize(c) for c in candidates]
        for col in df.columns:
            col_norm = _normalize(col)
            for cn in cand_norms:
                if cn and cn in col_norm:
                    return col
        return None

    # 1) Age
    if 'Age' in df.columns:
        age_col = 'Age'
    else:
        age_col = locate(age_candidates, substr_fallback=['age', 'year', 'yr'])
        if age_col is None:
            raise ValueError("Input dataframe must contain 'Age' (or a recognizable equivalent, e.g. 'feature3').")
        if age_col != 'Age':
            df = df.rename(columns={age_col: 'Age'})
            age_col = 'Age'

    # Ensure Age numeric
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df = df.dropna(subset=['Age'])
    df['Age'] = df['Age'].astype(float)

    # 2) MajorityFirst
    if 'MajorityFirst' in df.columns:
        mf_col = 'MajorityFirst'
    else:
        mf_col = locate(majorityfirst_candidates, substr_fallback=['majorityfirst', 'majority', 'first'])
        if mf_col is None:
            raise ValueError("Input dataframe must contain 'MajorityFirst' (or a recognizable equivalent, e.g. 'feature4').")
        if mf_col != 'MajorityFirst':
            df = df.rename(columns={mf_col: 'MajorityFirst'})
            mf_col = 'MajorityFirst'

    # Normalize MajorityFirst to 0/1 integers
    mf_raw = df['MajorityFirst']
    if pd.api.types.is_numeric_dtype(mf_raw):
        df['MajorityFirst'] = (mf_raw != 0).astype(int)
    else:
        mf_s = mf_raw.astype(str).str.strip().str.lower()
        df['MajorityFirst'] = mf_s.isin(['1', 'true', 't', 'yes', 'y', 'first', 'majorityfirst', 'mf', 'majority']).astype(int)

    df = df.dropna(subset=['MajorityFirst'])
    df['MajorityFirst'] = df['MajorityFirst'].astype(int)

    # 3) Site
    if 'Site' in df.columns:
        site_col = 'Site'
    else:
        site_col = locate(site_candidates, substr_fallback=['site', 'location', 'country', 'context'])
        if site_col is None:
            heuristics = ['loc', 'place', 'country', 'site', 'countryof', 'country_']
            site_col = locate(heuristics, substr_fallback=None)
        if site_col is None:
            raise ValueError("Input dataframe must contain 'Site' (or a recognizable equivalent, e.g. 'feature5').")
        if site_col != 'Site':
            df = df.rename(columns={site_col: 'Site'})
            site_col = 'Site'

    df['Site'] = df['Site'].astype('category')
    df = df.dropna(subset=['Site'])

    # 4) IsBoy (preferred) or derive from Gender
    if 'IsBoy' in df.columns:
        isboy_raw = df['IsBoy']
        if pd.api.types.is_numeric_dtype(isboy_raw):
            df['IsBoy'] = (isboy_raw != 0).astype(int)
        else:
            s = isboy_raw.astype(str).str.strip().str.lower()
            df['IsBoy'] = s.isin(['1', 'true', 't', 'yes', 'y', 'male', 'm', 'boy', 'b']).astype(int)
        df = df.dropna(subset=['IsBoy'])
        df['IsBoy'] = df['IsBoy'].astype(int)
    else:
        gender_col = locate(gender_candidates, substr_fallback=['gender', 'sex'])
        if gender_col is None:
            raise ValueError("Input dataframe must contain 'Gender' (or 'feature2') or 'IsBoy'.")
        if gender_col != 'Gender':
            df = df.rename(columns={gender_col: 'Gender'})
            gender_col = 'Gender'

        gen_raw = df['Gender']
        gen_num = pd.to_numeric(gen_raw, errors='coerce')
        if gen_num.notna().any():
            df['IsBoy'] = np.where(gen_num == 2, 1, np.where(gen_num.notna(), 0, np.nan))
            if df['IsBoy'].isna().any():
                s = gen_raw.astype(str).str.strip().str.lower()
                mapped = s.isin(['2', 'male', 'm', 'boy', 'b', 'man']).astype(float)
                df.loc[df['IsBoy'].isna(), 'IsBoy'] = mapped[df['IsBoy'].isna()]
        else:
            s = gen_raw.astype(str).str.strip().str.lower()
            df['IsBoy'] = s.isin(['2', 'male', 'm', 'boy', 'b', 'man']).astype(int)

        df = df.dropna(subset=['IsBoy'])
        df['IsBoy'] = df['IsBoy'].astype(int)

    # 5) MajorityChoice or derive from Choice (or other majority-indicating columns)
    if 'MajorityChoice' in df.columns:
        mc_raw = df['MajorityChoice']
        if pd.api.types.is_numeric_dtype(mc_raw):
            df['MajorityChoice'] = (mc_raw != 0).astype(int)
        else:
            s = mc_raw.astype(str).str.strip().str.lower()
            df['MajorityChoice'] = s.isin(['1', 'true', 't', 'yes', 'y', 'majority']).astype(int)
        df = df.dropna(subset=['MajorityChoice'])
        df['MajorityChoice'] = df['MajorityChoice'].astype(int)
    else:
        choice_col = locate(choice_candidates, substr_fallback=['choice', 'response', 'selection'])
        if choice_col is not None:
            if choice_col != 'Choice':
                df = df.rename(columns={choice_col: 'Choice'})
                choice_col = 'Choice'

            choice_raw = df['Choice']
            choice_num = pd.to_numeric(choice_raw, errors='coerce')
            if choice_num.notna().any():
                df['MajorityChoice'] = (choice_num == 2).astype(int)
            else:
                s = choice_raw.astype(str).str.strip().str.lower()
                df['MajorityChoice'] = s.isin(['2', 'majority', 'maj']).astype(int)

            df = df.dropna(subset=['MajorityChoice'])
            df['MajorityChoice'] = df['MajorityChoice'].astype(int)
        else:
            # Try to find any column that likely encodes whether majority was chosen directly
            maj_like = _find_column_by_substring(df, ['majority', 'maj', 'major', 'majorchoice', 'majoritychoice', 'majority_chosen', 'majoritychosen'])
            if maj_like is not None:
                mc_raw = df[maj_like]
                if pd.api.types.is_numeric_dtype(mc_raw):
                    df['MajorityChoice'] = (mc_raw != 0).astype(int)
                else:
                    s = mc_raw.astype(str).str.strip().str.lower()
                    df['MajorityChoice'] = s.isin(['1', 'true', 't', 'yes', 'y', 'majority', 'maj', 'chosen']).astype(int)
                df = df.dropna(subset=['MajorityChoice'])
                df['MajorityChoice'] = df['MajorityChoice'].astype(int)
            else:
                # As an additional fallback, search any column names for common tokens and attempt mapping
                fallback_tokens = ['choice', 'select', 'pick', 'resp', 'response', 'selection', 'chosen', 'option']
                fallback_col = _find_column_by_substring(df, fallback_tokens)
                if fallback_col is not None:
                    # Attempt to interpret fallback_col similarly to Choice
                    fc_raw = df[fallback_col]
                    fc_num = pd.to_numeric(fc_raw, errors='coerce')
                    if fc_num.notna().any():
                        # If numeric and values are 0/1, assume 1 means majority chosen; else if values 1/2/3 use 2 as majority
                        unique_vals = pd.Series(fc_num.dropna().unique())
                        if set(unique_vals).issubset({0, 1}):
                            df['MajorityChoice'] = (fc_num == 1).astype(int)
                        else:
                            df['MajorityChoice'] = (fc_num == 2).astype(int)
                    else:
                        s = fc_raw.astype(str).str.strip().str.lower()
                        df['MajorityChoice'] = s.isin(['1', '2', 'majority', 'maj', 'chosen']).astype(int)
                    df = df.dropna(subset=['MajorityChoice'])
                    df['MajorityChoice'] = df['MajorityChoice'].astype(int)
                else:
                    raise ValueError("Input dataframe must contain 'Choice' (or 'feature1') or 'MajorityChoice'.")

    # Mean-center age
    df['Age_c'] = df['Age'] - df['Age'].mean()

    # Final check that all required final columns are present
    required_final = ['Age', 'Age_c', 'MajorityChoice', 'Site', 'IsBoy', 'MajorityFirst']
    missing_final = [c for c in required_final if c not in df.columns]
    if missing_final:
        raise ValueError(f"Transform failed to produce required final columns: {missing_final}")

    # Ensure correct dtypes: Age float, Age_c float, MajorityChoice int, Site category, IsBoy int, MajorityFirst int
    df['Age'] = df['Age'].astype(float)
    df['Age_c'] = df['Age_c'].astype(float)
    df['MajorityChoice'] = df['MajorityChoice'].astype(int)
    if not pd.api.types.is_categorical_dtype(df['Site']):
        df['Site'] = df['Site'].astype('category')
    df['IsBoy'] = df['IsBoy'].astype(int)
    df['MajorityFirst'] = df['MajorityFirst'].astype(int)

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (binomial GLM) predicting the probability of choosing the majority option.
    Formula:
      MajorityChoice ~ Age_c * C(Site) + IsBoy + MajorityFirst

    Returns the fitted statsmodels GLMResults object. Attaches marginal effects (if computable)
    as results._marginal_effects for convenience.
    """
    # Required inputs for model
    required_cols = ['MajorityChoice', 'Age_c', 'Site', 'IsBoy', 'MajorityFirst']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    formula = 'MajorityChoice ~ Age_c * C(Site) + IsBoy + MajorityFirst'
    results = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Attach marginal effects if possible
    try:
        marg_eff = results.get_margeff(at='overall', method='dydx', dummy=True)
        results._marginal_effects = marg_eff
    except Exception:
        results._marginal_effects = None

    return results