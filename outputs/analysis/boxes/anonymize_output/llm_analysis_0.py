from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Attempt to read example CSV if available, but don't fail import if not.
try:
    df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/anonymize_output/boxes.csv')
except Exception:
    df = pd.DataFrame()


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe suitable for modeling.

    Input columns (expected in raw df) are attempted to be found among several
    common alternative names. The transform will create canonical columns
    'feature1'..'feature5' if alternatives exist. If none of the alternatives
    are present for a particular feature, the canonical column will be created
    with NA values and later rows with missing values for required inputs will
    be dropped.

    Output (added/derived) columns used in modeling:
      - MajorityChosen: binary outcome (1 if feature1 == 2, else 0)
      - Age: original age as float
      - Age_centered: Age centered at sample mean
      - Age_sq: squared centered age to capture quadratic effects
      - Female: 1 if girl (feature2 == 1), else 0
      - MajorityFirst: binary indicator from feature4 (coerced to 0/1)
      - Site: categorical site label, e.g., 'Site_1', 'Site_2', ...
    """
    # Work on a copy
    df = df.copy()

    # Define candidate alternative names for each expected raw-feature column.
    candidate_map = {
        'feature1': ['feature1', 'feature_1', 'feat1', 'outcome', 'choice', 'selected_option', 'response'],
        'feature2': ['feature2', 'feature_2', 'gender', 'sex'],
        'feature3': ['feature3', 'feature_3', 'age', 'Age', 'child_age', 'childage'],
        'feature4': ['feature4', 'feature_4', 'majority_first', 'demonstration_first', 'majorityfirst', 'order_first'],
        'feature5': ['feature5', 'feature_5', 'site', 'site_id', 'siteid', 'Site', 'location', 'siteId']
    }

    # For each canonical feature name, if any candidate exists in df, copy it to the canonical name.
    # If none exist, create the canonical column filled with NA so subsequent operations don't KeyError.
    for canonical, candidates in candidate_map.items():
        found = None
        for cand in candidates:
            if cand in df.columns:
                found = cand
                break
        if found is not None:
            # Copy column to canonical name if it's not already the canonical name
            if found != canonical:
                df[canonical] = df[found].copy()
            # if found == canonical, leave as is
        else:
            # Create the canonical column with NA so that dropna(subset=...) won't KeyError
            # Preserve index (even if empty)
            df[canonical] = pd.Series(np.nan, index=df.index)

    # Now drop rows with missing values in any of the key variables (after ensuring canonical cols exist)
    # feature5 is required for Site; if missing, rows are dropped.
    required_cols = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
    # Only perform dropna if there are rows; for empty df this is a no-op and preserves columns
    if df.shape[0] > 0:
        df = df.dropna(subset=required_cols)

    # Ensure types are numeric where appropriate
    # Use to_numeric with errors='coerce' which will produce NaN for bad conversions
    df['feature1'] = pd.to_numeric(df['feature1'], errors='coerce')
    df['feature2'] = pd.to_numeric(df['feature2'], errors='coerce')
    df['feature3'] = pd.to_numeric(df['feature3'], errors='coerce')
    df['feature4'] = pd.to_numeric(df['feature4'], errors='coerce')
    # feature5 may be non-numeric site ids; keep as string-safe representation but attempt numeric conversion too
    # Use combine_first to prefer numeric if successful, else original strings
    feature5_num = pd.to_numeric(df['feature5'], errors='coerce')
    df['feature5'] = feature5_num.combine_first(df['feature5'])

    # Re-drop if conversion introduced NA in the truly required numeric columns (1-4)
    if df.shape[0] > 0:
        df = df.dropna(subset=['feature1', 'feature2', 'feature3', 'feature4'])

    # Dependent variable: 1 if child chose the majority option (feature1 == 2), else 0
    # If df is empty, this will create an empty column
    df['MajorityChosen'] = (df['feature1'] == 2).astype(int)

    # Age and polynomial terms
    df['Age'] = df['feature3'].astype(float)
    # If no rows remain, avoid NaN mean by leaving Age_centered/Age_sq as NaN (but columns must exist)
    if len(df) > 0:
        df['Age_centered'] = df['Age'] - df['Age'].mean()
        df['Age_sq'] = df['Age_centered'] ** 2
    else:
        df['Age_centered'] = pd.Series(dtype=float, index=df.index)
        df['Age_sq'] = pd.Series(dtype=float, index=df.index)

    # Gender control: Female = 1 if feature2 == 1 (girl), else 0
    df['Female'] = pd.to_numeric(df['feature2'], errors='coerce').fillna(0).astype(int)
    df['Female'] = (df['Female'] == 1).astype(int)

    # Presentation order control: MajorityFirst (coerce to 0/1)
    df['MajorityFirst'] = pd.to_numeric(df['feature4'], errors='coerce').fillna(0).astype(int)
    df['MajorityFirst'] = (df['MajorityFirst'] == 1).astype(int)

    # Site as categorical moderator.
    # Create string labels to make grouping/printing clearer.
    # Preserve original feature5 values where numeric; otherwise use their string form.
    feature5_series = df['feature5'].astype(str).str.strip()
    # Remove possible trailing ".0" from numeric conversions
    feature5_series = feature5_series.str.replace(r'\.0$', '', regex=True)
    df['Site'] = 'Site_' + feature5_series
    df['Site'] = df['Site'].astype('category')

    # Keep only columns we will use (plus originals for reference)
    keep_cols = [
        'feature1', 'feature2', 'feature3', 'feature4', 'feature5',
        'MajorityChosen', 'Age', 'Age_centered', 'Age_sq', 'Female', 'MajorityFirst', 'Site'
    ]

    # Ensure we only return columns that exist in df
    keep_cols_existing = [c for c in keep_cols if c in df.columns]

    # Return final transformed dataframe (may be empty but must contain required columns)
    return df[keep_cols_existing]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression predicting reliance on the majority (MajorityChosen)
    from Age (linear + quadratic), with Site as a categorical moderator on Age
    (Age x Site interactions). Controls: Female and MajorityFirst. Uses cluster-robust
    standard errors clustered by Site when multiple sites are present.

    Returns:
      - results: the fitted GLM results object (Binomial family) with clustered SEs when possible,
                 or None if no fitting was performed (e.g., empty input).
      - predictions_df: dataframe with predicted probabilities and key columns
    """
    # Ensure the required columns exist
    required = ['MajorityChosen', 'Age_centered', 'Age_sq', 'Female', 'MajorityFirst', 'Site', 'Age']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns in transformed df: {missing}")

    # If there is no data to fit, return an empty predictions dataframe and no results
    if df.shape[0] == 0:
        preds = pd.DataFrame(columns=['MajorityChosen', 'pred_prob', 'Age', 'Site', 'Female', 'MajorityFirst'])
        # Ensure Site is categorical type even if empty
        preds['Site'] = preds['Site'].astype('category')
        return {
            'results': None,
            'predictions_df': preds
        }

    # Determine number of observed Site levels (exclude NA)
    n_sites = int(df['Site'].nunique(dropna=True))

    if n_sites == 0:
        raise ValueError("No observed Site levels found in the data; cannot model Site.")
    # Build formula depending on whether multiple sites exist
    if n_sites >= 2:
        # Full model with Site and Age x Site interaction
        formula = 'MajorityChosen ~ Age_centered + Age_sq + Female + MajorityFirst + C(Site) + Age_centered:C(Site)'
    else:
        # Only one site present: cannot include C(Site) or interactions
        formula = 'MajorityChosen ~ Age_centered + Age_sq + Female + MajorityFirst'

    # Fit a binomial GLM (logit link).
    model_glm = smf.glm(formula, data=df, family=sm.families.Binomial())

    # Fit with cluster-robust standard errors (cluster by Site) only when multiple clusters exist
    if n_sites >= 2:
        try:
            results = model_glm.fit(cov_type='cluster', cov_kwds={'groups': df['Site']})
        except Exception:
            # Fall back to default fit if clustering fails
            results = model_glm.fit()
    else:
        # Single site: do a regular fit
        results = model_glm.fit()

    # As a helpful additional object, supply predicted probabilities for each observation
    df = df.copy()
    try:
        df['pred_prob'] = results.predict(df)
    except Exception:
        df['pred_prob'] = np.nan

    # Return a dictionary with the results object and the dataframe with predictions
    predictions_df = df[['MajorityChosen', 'pred_prob', 'Age', 'Site', 'Female', 'MajorityFirst']].copy()
    return {
        'results': results,
        'predictions_df': predictions_df
    }