from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/shuffle_names_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and derive variables for modeling the effect of children on extramarital affairs.

    Produces the following columns required by the model:
      - AffairCount: numeric count/frequency of affairs (coerced to numeric). If non-numeric labels such as 'none' are present they are mapped to 0.
      - AnyAffair: binary indicator 1 if AffairCount > 0 else 0
      - LogAffairCount: log(1 + AffairCount)
      - HasChildren: binary indicator 1 if there are children in the marriage, 0 if not
      - IsMale: binary indicator 1 = male, 0 = female (attempts to parse textual or numeric encodings)
      - Age: numeric age (from 'rating' column which appears to be age midpoint values)
      - Education: numeric education
      - Religiousness: numeric religiousness
      - YearsMarried: numeric years married
      - MaritalHappiness: numeric marriage happiness (rownames column)

    The function attempts robust parsing when the dataset uses variant encodings (e.g., 'yes'/'no', 'Y'/'N', 1/0, 'male'/'female'). Rows with missing DV (AffairCount) or missing IV (HasChildren) after parsing are dropped.
    """
    df = df.copy()

    # --- AffairCount ---
    # Standardize common textual entries, then coerce to numeric
    if 'affairs' not in df.columns:
        raise KeyError("Expected column 'affairs' in the input dataframe")

    # Replace common textual 'none' or similar with 0
    df['affairs'] = df['affairs'].replace({
        'none': 0, 'None': 0, 'NONE': 0, 'no': 0, 'No': 0, 'NA': np.nan, '': np.nan
    })
    # If there are fractional-like or other markers, attempt conversion; errors -> NaN
    df['AffairCount'] = pd.to_numeric(df['affairs'], errors='coerce')

    # If AffairCount appears to be a coded categorical (e.g., frequency codes), keep as numeric variable.
    # Create binary AnyAffair and continuous log transform for OLS specifications
    df['AnyAffair'] = (df['AffairCount'] > 0).astype(float)
    df['LogAffairCount'] = np.log1p(df['AffairCount'].fillna(0))

    # --- HasChildren (IV) ---
    # Attempt to find the appropriate column that indicates children. Common encodings: 'children', 'age' (sometimes mislabelled).
    # We try 'children' first, then 'age'.
    has_children_col = None
    if 'children' in df.columns:
        has_children_col = 'children'
    elif 'age' in df.columns:
        has_children_col = 'age'

    def parse_binary_flag(series: pd.Series) -> pd.Series:
        """Return series of {0,1,NaN} by heuristic parsing of values that mean yes/no or 1/0."""
        s = series.copy()
        # Normalize strings
        s = s.replace({True: '1', False: '0'})
        s = s.astype('object')
        s = s.str.strip().str.lower().replace({'t': '1', 'f': '0'})
        mapping = {
            'yes': 1, 'y': 1, '1': 1, 'true': 1, 'has children': 1,
            'no': 0, 'n': 0, '0': 0, 'false': 0
        }
        parsed = s.map(mapping)
        # If result is all NaN and series is numeric, try numeric mapping directly
        if parsed.isna().all():
            parsed = pd.to_numeric(series, errors='coerce')
            # Tentative rule: treat >0 as has children
            parsed = parsed.apply(lambda x: 1 if pd.notna(x) and x > 0 else (0 if pd.notna(x) and x == 0 else np.nan))
        return parsed.astype(float)

    if has_children_col is not None:
        df['HasChildren'] = parse_binary_flag(df[has_children_col])
    else:
        # If no plausible column, create NaN column so model can detect missingness
        df['HasChildren'] = np.nan

    # --- Gender ---
    # Create IsMale from 'gender' column if present, otherwise attempt to infer from 'children' column (in case of mislabelled schema)
    def parse_gender(series: pd.Series) -> pd.Series:
        s = series.copy().astype('object')
        s = s.str.strip().str.lower()
        mapping = {'male': 1, 'm': 1, 'man': 1, 'female': 0, 'f': 0, 'woman': 0}
        parsed = s.map(mapping)
        if parsed.isna().all():
            # try numeric: assume values where >0.5 indicates male in some codings, otherwise try common numeric codes
            num = pd.to_numeric(series, errors='coerce')
            if not num.isna().all():
                # Heuristic: if unique values are {0,1} or {1,2}, map accordingly.
                uniq = num.dropna().unique()
                if set(np.unique(uniq)).issubset({0, 1}):
                    parsed = num.apply(lambda x: 1 if x == 1 else (0 if x == 0 else np.nan))
                elif set(np.unique(uniq)).issubset({1, 2}):
                    # common encoding 1=male,2=female or vice-versa — assume 1=male
                    parsed = num.apply(lambda x: 1 if x == 1 else (0 if x == 2 else np.nan))
                else:
                    # fallback: treat values > 1.5 as male
                    parsed = num.apply(lambda x: 1 if pd.notna(x) and x > 1.5 else (0 if pd.notna(x) and x <= 1.5 else np.nan))
        return parsed.astype(float)

    if 'gender' in df.columns:
        df['IsMale'] = parse_gender(df['gender'])
    else:
        df['IsMale'] = np.nan

    # --- Controls: Age, Education, Religiousness, YearsMarried, MaritalHappiness ---
    # Age comes from 'rating' column in the supplied schema (this column appears to encode age midpoints)
    if 'rating' in df.columns:
        df['Age'] = pd.to_numeric(df['rating'], errors='coerce')
    elif 'age' in df.columns:
        # if rating absent but age exists, try to parse it
        df['Age'] = pd.to_numeric(df['age'], errors='coerce')
    else:
        df['Age'] = np.nan

    if 'education' in df.columns:
        df['Education'] = pd.to_numeric(df['education'], errors='coerce')
    else:
        df['Education'] = np.nan

    if 'religiousness' in df.columns:
        df['Religiousness'] = pd.to_numeric(df['religiousness'], errors='coerce')
    else:
        df['Religiousness'] = np.nan

    if 'yearsmarried' in df.columns:
        df['YearsMarried'] = pd.to_numeric(df['yearsmarried'], errors='coerce')
    else:
        df['YearsMarried'] = np.nan

    # 'rownames' in the schema appears to encode self-rating of marriage
    if 'rownames' in df.columns:
        df['MaritalHappiness'] = pd.to_numeric(df['rownames'], errors='coerce')
    else:
        df['MaritalHappiness'] = np.nan

    # --- Final cleaning: drop rows missing the main IV or DV ---
    # We need HasChildren and AffairCount for model. Drop rows where either is missing.
    df_model = df.dropna(subset=['HasChildren', 'AffairCount'])

    # Reset index for cleanliness
    df_model = df_model.reset_index(drop=True)

    # Return the full transformed dataframe (with original columns preserved) but guaranteed to include
    # the derived columns used in modeling.
    return df_model


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit multiple specifications to estimate the relationship between having children and extramarital affairs.

    Models fitted:
      1) OLS on log(1 + AffairCount) to account for skewness (dependent variable = LogAffairCount)
      2) Poisson GLM on raw AffairCount (counts)
      3) Negative Binomial GLM on raw AffairCount (to allow overdispersion)

    All specifications include the same controls: Age, IsMale, Education, Religiousness, YearsMarried, MaritalHappiness.

    Returns a dict with fitted results objects for each model.
    """
    results = {}

    # Ensure required columns exist
    required = ['AffairCount', 'LogAffairCount', 'HasChildren', 'Age', 'IsMale', 'Education', 'Religiousness', 'YearsMarried', 'MaritalHappiness']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"The transformed dataframe is missing required columns: {missing}")

    # Define model matrix: drop rows with missing controls
    model_df = df.copy()
    model_df = model_df.dropna(subset=['HasChildren', 'AffairCount'])
    # For controls, allow rows with any missing control to be dropped (listwise deletion)
    model_df = model_df.dropna(subset=['Age', 'IsMale', 'Education', 'Religiousness', 'YearsMarried', 'MaritalHappiness'])

    # Predictor columns
    predictors = ['HasChildren', 'Age', 'IsMale', 'Education', 'Religiousness', 'YearsMarried', 'MaritalHappiness']

    # Prepare X and add constant
    X = model_df[predictors]
    X = sm.add_constant(X, has_constant='add')

    # 1) OLS on LogAffairCount
    y_ols = model_df['LogAffairCount']
    ols_model = sm.OLS(y_ols, X).fit()
    results['ols_log1p'] = ols_model

    # 2) Poisson GLM on AffairCount
    y_count = model_df['AffairCount']
    # Use GLM Poisson; supply exposure if needed; here none
    poisson_model = sm.GLM(y_count, X, family=sm.families.Poisson()).fit()
    results['poisson'] = poisson_model

    # 3) Negative Binomial GLM to allow overdispersion
    try:
        nb_model = sm.GLM(y_count, X, family=sm.families.NegativeBinomial()).fit()
        results['neg_binomial'] = nb_model
    except Exception:
        # Fall back to discrete NegativeBinomial if GLM NegativeBinomial not available or fails
        try:
            from statsmodels.discrete.discrete_model import NegativeBinomial
            nb_model2 = NegativeBinomial(y_count, X).fit(disp=False)
            results['neg_binomial'] = nb_model2
        except Exception:
            results['neg_binomial'] = None

    # Return the fitted models. Each entry is the statsmodels results instance (use .summary() outside if desired).
    return results


