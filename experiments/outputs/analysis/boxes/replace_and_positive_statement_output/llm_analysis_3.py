from typing import Any, Dict, List
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Note: top-level CSV read in the original file is left out to keep this module
# focused on the transform() and model() functions. The functions operate on
# DataFrame inputs and satisfy the required column-name contract.


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the analysis-ready dataframe.

    Creates and preserves required final columns:
      - 'y' (unchanged except for type enforcement)
      - 'MajorityChosen': binary indicator for choosing the majority option (y == 2)
      - 'is_girl': binary indicator (1 if gender == 1, else 0)
      - 'age_centered': age minus mean(age) (float)
      - 'age_group': categorical age bins (4-6, 7-9, 10-12, 13-14)
      - 'culture_cat': string categorical for culture (e.g., 'C1', 'C2', ...)
      - culture dummy columns: one-hot encoding with prefix 'culture_' (drop_first=True)
      - age-by-culture interaction columns: age_centered * culture_* for each culture dummy
      - 'y_mn': multinomial coding for MNLogit (0..K-1) derived from y (original y is 1..3)

    Returns the dataframe with all these columns appended.
    """
    df = df.copy()

    # Drop rows with missing essential variables
    df = df.dropna(subset=['y', 'age', 'culture', 'gender', 'majority_first'])

    # Ensure types for core columns
    # Keep original 'y' column name as required
    df['y'] = pd.to_numeric(df['y'], errors='coerce').astype(int)
    df['age'] = pd.to_numeric(df['age'], errors='coerce').astype(float)
    # culture may be numeric site id; cast to string-safe int where possible for category labels
    df['culture'] = pd.to_numeric(df['culture'], errors='coerce').astype(int)
    df['gender'] = pd.to_numeric(df['gender'], errors='coerce').astype(int)
    df['majority_first'] = pd.to_numeric(df['majority_first'], errors='coerce').astype(int)

    # Dependent-variable-derived columns
    df['MajorityChosen'] = (df['y'] == 2).astype(int)

    # Control: gender mapped to is_girl (preserve required column name)
    df['is_girl'] = (df['gender'] == 1).astype(int)

    # Age (centered) and age groups (developmental stages)
    df['age_centered'] = df['age'] - df['age'].mean()
    df['age_centered'] = df['age_centered'].astype(float)
    # Keep 'age_group' column as required
    df['age_group'] = pd.cut(df['age'], bins=[3, 6, 9, 12, 15],
                             labels=['4-6', '7-9', '10-12', '13-14'], right=True)

    # Culture as categorical string for later one-hot encoding
    df['culture_cat'] = 'C' + df['culture'].astype(str)

    # One-hot encode culture (drop first to avoid perfect multicollinearity)
    culture_dummies = pd.get_dummies(df['culture_cat'], prefix='culture', drop_first=True)
    # Ensure dummy columns are numeric (0/1)
    culture_dummies = culture_dummies.astype(int)

    # Attach dummies to df
    df = pd.concat([df, culture_dummies], axis=1)

    # Use the list of dummy columns we just created (avoid scanning dataframe which might catch other similarly-named cols)
    culture_dummy_cols = culture_dummies.columns.tolist()

    # Create age-by-culture interaction columns (safe numeric coercion)
    for c in culture_dummy_cols:
        # Ensure the dummy column is numeric and has no problematic types
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(float)
        df[f'age_x_{c}'] = df['age_centered'] * df[c]

    # Prepare a version of y for statsmodels MNLogit which expects 0..K-1
    # Original y: 1 = unchosen/undemonstrated, 2 = majority, 3 = minority
    df['y_mn'] = (df['y'] - 1).astype(int)

    # Save lists of derived columns in attributes for reproducibility (optional)
    df.attrs['culture_dummy_cols'] = culture_dummy_cols
    df.attrs['interaction_cols'] = [f'age_x_{c}' for c in culture_dummy_cols]

    return df


def model(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Fit two complementary models to test whether children's reliance on social information
    and preference for majority cues vary across cultures and developmental stages.

    Models fitted:
      1. Multinomial logistic regression (MNLogit) predicting the 3-choice outcome y
         (unchosen / majority / minority) from age (centered), culture (dummies), the
         age x culture interactions, and controls (is_girl, majority_first).

      2. Binary logistic regression (Logit) predicting MajorityChosen (1 if majority chosen)
         with the same predictors.

    Returns a dict with fitted model objects and textual summaries or error messages.
    """
    results: Dict[str, Any] = {}

    # Ensure the columns we rely on exist
    if 'y_mn' not in df.columns:
        raise ValueError("Expected column 'y_mn' to be present. Run transform() first.")

    # Build predictor list programmatically based on transform outputs
    base_predictors = ['age_centered', 'is_girl', 'majority_first']

    # Prefer the attrs saved by transform; otherwise, derive safely from df columns
    culture_dummy_cols = df.attrs.get(
        'culture_dummy_cols',
        [c for c in df.columns if c.startswith('culture_')]
    )
    # Ensure the listed culture dummies actually exist in the dataframe
    culture_dummy_cols = [c for c in culture_dummy_cols if c in df.columns]

    interaction_cols = df.attrs.get(
        'interaction_cols',
        [f'age_x_{c}' for c in culture_dummy_cols]
    )
    interaction_cols = [c for c in interaction_cols if c in df.columns]

    exog_cols = base_predictors + culture_dummy_cols + interaction_cols

    # Filter exog_cols to those actually present (defensive)
    exog_cols = [c for c in exog_cols if c in df.columns]

    # Prepare exog and coerce to numeric; fillna with 0.0 for predictors (common safe practice)
    exog = df[exog_cols].copy()
    exog = exog.apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(float)
    exog = sm.add_constant(exog, has_constant='add')

    # 1) Multinomial logistic regression (predicting 3 categories)
    endog_multi = pd.to_numeric(df['y_mn'], errors='coerce').astype(int)

    try:
        mnlogit_model = sm.MNLogit(endog_multi, exog)
        mnlogit_fit = mnlogit_model.fit(method='newton', maxiter=200, disp=False)
        results['mnlogit_fit'] = mnlogit_fit
        results['mnlogit_summary'] = mnlogit_fit.summary().as_text()
    except Exception as e:
        results['mnlogit_error'] = str(e)

    # 2) Binary logistic regression for majority preference
    endog_bin = pd.to_numeric(df['MajorityChosen'], errors='coerce').fillna(0).astype(int)
    try:
        logit_model = sm.Logit(endog_bin, exog)
        logit_fit = logit_model.fit(disp=False)
        results['logit_majority_fit'] = logit_fit
        results['logit_majority_summary'] = logit_fit.summary().as_text()
    except Exception as e:
        results['logit_majority_error'] = str(e)

    # Also return the list of exogenous columns used so downstream code knows what was fit
    results['exog_cols'] = exog.columns.tolist()

    return results