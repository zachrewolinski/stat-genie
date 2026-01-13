from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Attempt to read a default CSV if available; fail silently to keep module import-safe.
try:
    _df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/anonymize_output/boxes.csv')
except Exception:
    _df = None


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the analysis dataframe.

    Produces the following columns used by the model:
      - IsMajority: binary DV (1 if feature1 == 2, else 0)
      - Age: numeric age in years (from feature3)
      - Age_centered: Age centered at the sample mean
      - Age_centered_sq: squared centered age (to capture nonlinearity)
      - Female: gender control (1 if feature2 == 1 (girl), 0 if boy)
      - MajorityFirst: whether majority option was demonstrated first (from feature4, coerced to 0/1)
      - Site: categorical site label constructed from feature5 (e.g., 'Site_1')

    Drops rows with missing values in the core fields.
    """
    df = df.copy()

    # If the dataframe already contains the final required columns, try to standardize/complete them.
    final_cols = ['IsMajority', 'Age', 'Age_centered', 'Age_centered_sq', 'Female', 'MajorityFirst', 'Site']
    if set(final_cols).issubset(df.columns):
        # Ensure Age is numeric
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

        # Compute centered age and quadratic if missing or inconsistent
        if 'Age_centered' not in df.columns or df['Age_centered'].isnull().all():
            df['Age_centered'] = df['Age'] - df['Age'].mean()
        if 'Age_centered_sq' not in df.columns or df['Age_centered_sq'].isnull().all():
            df['Age_centered_sq'] = df['Age_centered'] ** 2

        # Ensure binary integer types for controls
        df['IsMajority'] = pd.to_numeric(df['IsMajority'], errors='coerce').fillna(0).astype(int)
        df['Female'] = pd.to_numeric(df['Female'], errors='coerce').fillna(0).astype(int)
        df['MajorityFirst'] = pd.to_numeric(df['MajorityFirst'], errors='coerce').fillna(0).astype(int)

        # Ensure Site is categorical in the expected string form
        df['Site'] = df['Site'].astype(str)
        if not df['Site'].str.startswith('Site_').all():
            df['Site'] = 'Site_' + df['Site']
        df['Site'] = df['Site'].astype('category')

        df_out = df.loc[:, final_cols].reset_index(drop=True)
        return df_out

    # If raw feature columns are present, use them
    raw_required = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
    if set(raw_required).issubset(df.columns):
        # Drop rows missing essential raw fields
        df = df.dropna(subset=raw_required)

        # Dependent variable: did the child choose the majority option? (feature1: 2 = majority option)
        df['IsMajority'] = (pd.to_numeric(df['feature1'], errors='coerce') == 2).astype(int)

        # Age: numeric from feature3
        df['Age'] = pd.to_numeric(df['feature3'], errors='coerce')

        # Center age and add quadratic term
        df['Age_centered'] = df['Age'] - df['Age'].mean()
        df['Age_centered_sq'] = df['Age_centered'] ** 2

        # Gender: feature2 (1 = girl, 2 = boy) -> Female: 1 if girl else 0
        df['Female'] = (pd.to_numeric(df['feature2'], errors='coerce') == 1).astype(int)

        # Majority-first indicator: coerce to numeric 0/1
        df['MajorityFirst'] = (pd.to_numeric(df['feature4'], errors='coerce') == 1).astype(int)

        # Site: make a categorical label. Keep the numeric id but make explicit category string.
        df['Site'] = pd.to_numeric(df['feature5'], errors='coerce')
        # If conversion produced NaNs, fallback to string-preserved site ids
        if df['Site'].isnull().any():
            df['Site'] = 'Site_' + df['feature5'].astype(str)
        else:
            df['Site'] = 'Site_' + df['Site'].astype(int).astype(str)
        df['Site'] = df['Site'].astype('category')

        out_cols = ['IsMajority', 'Age', 'Age_centered', 'Age_centered_sq', 'Female', 'MajorityFirst', 'Site']
        df_out = df.loc[:, out_cols].reset_index(drop=True)
        return df_out

    # Attempt to infer mapping from common alternative column names (case-insensitive).
    # This is a best-effort fallback to make the function robust to different input schemas.
    col_map = {c.lower(): c for c in df.columns}
    mapped = {}

    def find_candidate(keys):
        for k in keys:
            if k in col_map:
                return col_map[k]
        return None

    mapped['Age'] = find_candidate(['age', 'child_age', 'age_years'])
    mapped['Female'] = find_candidate(['female', 'sex', 'gender', 'is_female'])
    # Include short/typical names like 'y' and 'response' for outcome columns
    mapped['IsMajority'] = find_candidate(['ismajority', 'is_majority', 'majority', 'choice', 'response', 'selected_majority', 'selected', 'y'])
    mapped['MajorityFirst'] = find_candidate(['majorityfirst', 'majority_first', 'order', 'order_shown', 'first_shown', 'first'])
    mapped['Site'] = find_candidate(['site', 'site_id', 'location', 'country', 'culture'])

    # If we can map all required conceptual variables, build the final dataframe.
    required_conceptual = ['IsMajority', 'Age', 'Female', 'MajorityFirst', 'Site']
    if all(mapped.get(k) for k in required_conceptual):
        # Coerce and compute
        df['IsMajority'] = pd.to_numeric(df[mapped['IsMajority']], errors='coerce').fillna(0).astype(int)
        df['Age'] = pd.to_numeric(df[mapped['Age']], errors='coerce')
        df['Age_centered'] = df['Age'] - df['Age'].mean()
        df['Age_centered_sq'] = df['Age_centered'] ** 2

        # Female: try numeric mapping where 1 = girl, otherwise map common string labels
        female_col = mapped['Female']
        female_numeric = pd.to_numeric(df[female_col], errors='coerce')
        if female_numeric.notna().any():
            df['Female'] = (female_numeric == 1).astype(int)
        else:
            # string-based mapping
            cond = df[female_col].astype(str).str.lower().isin(['female', 'f', 'girl', 'girl '])
            df['Female'] = cond.astype(int)

        # MajorityFirst: coerce to numeric where 1 indicates majority shown first
        df['MajorityFirst'] = (pd.to_numeric(df[mapped['MajorityFirst']], errors='coerce') == 1).astype(int)

        # Site: canonicalize to 'Site_<id>' strings
        df['Site'] = df[mapped['Site']].astype(str)
        if not df['Site'].str.startswith('Site_').all():
            df['Site'] = 'Site_' + df['Site']
        df['Site'] = df['Site'].astype('category')

        out_cols = ['IsMajority', 'Age', 'Age_centered', 'Age_centered_sq', 'Female', 'MajorityFirst', 'Site']
        df_out = df.loc[:, out_cols].reset_index(drop=True)
        return df_out

    # If we reach this point, we cannot reliably produce the required final dataframe.
    missing = [c for c in raw_required if c not in df.columns]
    raise ValueError(
        "Input dataframe does not contain the expected raw feature columns "
        f"{raw_required} and could not be auto-mapped to the required conceptual "
        f"variables. Present columns: {list(df.columns)}. Missing raw columns: {missing}"
    )


def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic (binomial) generalized linear model predicting the probability of choosing the majority option.

    Primary test: interaction between centered age and site (Age_centered * C(Site)) to test whether the developmental slope
    for majority reliance differs across cultural contexts.

    Model formula:
      IsMajority ~ Age_centered * C(Site) + Age_centered_sq + Female + MajorityFirst

    Returns the fitted GLMResults object.
    """
    # Ensure required columns exist
    required = ['IsMajority', 'Age_centered', 'Age_centered_sq', 'Female', 'MajorityFirst', 'Site']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"The dataframe passed to model is missing required columns: {missing}")

    formula = 'IsMajority ~ Age_centered * C(Site) + Age_centered_sq + Female + MajorityFirst'
    glm_binom = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()
    return glm_binom