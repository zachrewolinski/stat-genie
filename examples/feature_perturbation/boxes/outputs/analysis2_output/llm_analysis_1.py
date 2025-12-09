from typing import Any
import re
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe ready for modeling.

    Produces these columns used by the model:
      - MajorityChoice: binary (1 = chose majority, 0 = otherwise)
      - Age: numeric (years)
      - Age_c: mean-centered age
      - Age_sq: squared centered age (Age_c ** 2)
      - Site: categorical site label (e.g. 'Site_1', 'Site_2', ...)
      - IsBoy: gender dummy (1 = boy, 0 = girl)
      - DemoFirst: whether majority was demonstrated first (0/1)
    """

    df = df.copy()

    # Expected mapping from raw variable names to final names.
    rename_map = {
        'feature1': 'Outcome',   # 1 = unchosen option, 2 = majority option, 3 = minority option
        'feature2': 'Gender',    # 1 = girl, 2 = boy
        'feature3': 'Age',       # age in years
        'feature4': 'DemoFirst', # whether majority option was demonstrated first (0/1)
        'feature5': 'Site'       # site ID (1..8)
    }

    # If final names already exist, do not overwrite them.
    already_final = {v for v in rename_map.values() if v in df.columns}

    # If any of the expected raw keys exist, rename those.
    raw_keys_present = [k for k in rename_map.keys() if k in df.columns]
    if raw_keys_present:
        filtered_map = {k: v for k, v in rename_map.items() if k in df.columns and v not in already_final}
        if filtered_map:
            df = df.rename(columns=filtered_map)
    else:
        # If none of the raw keys are present and the final keys are also not present,
        # attempt to map by position: assume first five columns correspond to feature1..feature5.
        finals_present = [v for v in rename_map.values() if v in df.columns]
        if len(finals_present) == 0 and df.shape[1] >= 5:
            first_cols = list(df.columns[:5])
            positional_map = {first_cols[i]: list(rename_map.values())[i] for i in range(5)}
            df = df.rename(columns=positional_map)

    # At this point, we expect the following columns to exist (at least to attempt processing):
    expected_after_rename = ['Outcome', 'Gender', 'Age', 'DemoFirst', 'Site']
    missing = [c for c in expected_after_rename if c not in df.columns]
    if missing:
        raise ValueError(
            "transform expected the raw dataframe to contain columns that could be mapped to "
            "'Outcome', 'Gender', 'Age', 'DemoFirst', and 'Site'. Missing after attempted "
            f"renaming/mapping: {missing}. Available columns: {list(df.columns)}"
        )

    # Normalize / cast Age
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

    # Normalize Site into canonical labels like 'Site_1', 'Site_2', ...
    def normalize_site(x):
        if pd.isna(x):
            return np.nan
        # If already in desired format
        s = str(x)
        if s.startswith('Site_'):
            return s
        # Try numeric conversion
        try:
            n = int(float(x))
            return f"Site_{n}"
        except Exception:
            m = re.search(r'(\d+)', s)
            if m:
                return f"Site_{int(m.group(1))}"
            # Fallback: use cleaned string (prefix to avoid clashes)
            cleaned = re.sub(r'\s+', '_', s.strip())
            if cleaned == '':
                return np.nan
            # If cleaned already has 'Site_' anywhere, keep it
            if cleaned.startswith('Site_'):
                return cleaned
            return f"Site_{cleaned}"

    df['Site'] = df['Site'].apply(normalize_site)
    # Cast to category
    df['Site'] = df['Site'].astype('category')

    # Create dependent variable: MajorityChoice (1 if chose majority option, else 0)
    # Outcome codes: 2 = majority
    df['MajorityChoice'] = (df['Outcome'] == 2).astype(int)

    # Create gender indicator: IsBoy (1 = boy, 0 = girl). Accept various encodings.
    def map_isboy(x):
        if pd.isna(x):
            return np.nan
        # numeric possibilities
        try:
            xi = int(float(x))
            if xi == 2:
                return 1
            if xi == 1:
                return 0
        except Exception:
            pass
        s = str(x).strip().lower()
        if s in {'2', 'boy', 'male', 'm', 'b'}:
            return 1
        if s in {'1', 'girl', 'female', 'f', 'g'}:
            return 0
        # boolean-like
        if s in {'true', 't'}:
            # ambiguous; treat True as 1 (boy) only if original coding implied that; safer to set NA
            return np.nan
        return np.nan

    df['IsBoy'] = df['Gender'].apply(map_isboy)

    # Ensure DemoFirst is 0/1 integer
    def map_demofirst(x):
        if pd.isna(x):
            return np.nan
        # numeric
        try:
            xi = int(float(x))
            if xi in (0, 1):
                return xi
        except Exception:
            pass
        s = str(x).strip().lower()
        if s in {'0', 'no', 'false', 'f'}:
            return 0
        if s in {'1', 'yes', 'true', 't'}:
            return 1
        return np.nan

    df['DemoFirst'] = df['DemoFirst'].apply(map_demofirst)

    # Drop rows with missing critical fields after normalization
    required_for_model = ['MajorityChoice', 'Age', 'Site', 'IsBoy', 'DemoFirst']
    df = df.dropna(subset=required_for_model).reset_index(drop=True)

    # Mean-center Age and add quadratic term
    df['Age_c'] = df['Age'] - df['Age'].mean()
    df['Age_sq'] = df['Age_c'] ** 2

    # Keep only the columns needed for modeling plus original useful fields
    cols_to_keep = ['Outcome', 'MajorityChoice', 'Age', 'Age_c', 'Age_sq', 'Site', 'IsBoy', 'DemoFirst']
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
    df = df[cols_to_keep].reset_index(drop=True)

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression predicting the probability the child chose the majority option.

    Model specification:
      MajorityChoice ~ Age_c + Age_sq + C(Site) + Age_c:C(Site) + IsBoy + DemoFirst

    Returns:
      A statsmodels results object with cluster-robust standard errors by Site (if available).
    """

    # Check that required columns exist
    required = ['MajorityChoice', 'Age_c', 'Age_sq', 'Site', 'IsBoy', 'DemoFirst']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"model expected the transformed dataframe to contain columns: {missing}")

    formula = 'MajorityChoice ~ Age_c + Age_sq + C(Site) + Age_c:C(Site) + IsBoy + DemoFirst'

    # Fit a logistic regression (binomial logit)
    logit_model = smf.logit(formula=formula, data=df)
    results = logit_model.fit(disp=False)

    # Attempt to compute cluster-robust SEs clustered by Site to account for within-site correlation
    try:
        results_clustered = results.get_robustcov_results(cov_type='cluster', groups=df['Site'])
    except Exception:
        # If clustering fails (e.g., if only one observation per cluster), return the original results
        results_clustered = results

    return results_clustered