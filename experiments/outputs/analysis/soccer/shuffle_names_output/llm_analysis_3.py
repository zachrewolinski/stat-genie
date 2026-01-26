from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the original dataframe into the analysis-ready dataframe.

    Steps:
    - Ensure rater columns present and numeric and construct SkinTone as the average of rater1 and rater2.
    - Define a binary DarkSkin indicator: include only clearly 'light' and 'dark' ratings (exclude ambiguous middle ratings) for the primary analysis. "Light" = average <= 0.4, "Dark" = average >= 0.6. Rows with ambiguous average in (0.4,0.6) will be dropped for the primary analysis.
    - Ensure redCards and refNum are numeric counts; drop rows with missing or zero exposure (refNum <= 0).
    - Create offset = log(refNum) for use in Poisson/Negative Binomial models.
    - Create a descriptive red_rate column (redCards / refNum).

    Final dataframe will contain the required columns:
      rater1, rater2, SkinTone, DarkSkin, redCards, refNum, offset, red_rate,
      leagueCountry, club, player, goals
    """
    df = df.copy()

    # Ensure rater columns exist and are numeric
    if 'rater1' not in df.columns or 'rater2' not in df.columns:
        raise KeyError("Expected columns 'rater1' and 'rater2' in dataframe")

    df['rater1'] = pd.to_numeric(df['rater1'], errors='coerce')
    df['rater2'] = pd.to_numeric(df['rater2'], errors='coerce')

    # Compute continuous skin-tone measure as the average of the two raters
    df['SkinTone'] = df[['rater1', 'rater2']].mean(axis=1)

    # Define categorical 'SkinCategory' and binary 'DarkSkin'
    def _skin_cat(x):
        if pd.isna(x):
            return pd.NA
        if x <= 0.4:
            return 'Light'
        if x >= 0.6:
            return 'Dark'
        return 'Ambiguous'

    df['SkinCategory'] = df['SkinTone'].apply(_skin_cat)
    df['DarkSkin'] = df['SkinCategory'].map({'Dark': 1, 'Light': 0})

    # Ensure redCards and refNum are present and numeric
    if 'redCards' not in df.columns:
        raise KeyError("Expected column 'redCards' in dataframe")
    if 'refNum' not in df.columns:
        raise KeyError("Expected column 'refNum' (dyad match count/exposure) in dataframe")

    df['redCards'] = pd.to_numeric(df['redCards'], errors='coerce')
    df['refNum'] = pd.to_numeric(df['refNum'], errors='coerce')

    # Drop rows with missing key values for outcome/exposure/skin measures
    df = df.dropna(subset=['redCards', 'refNum', 'SkinTone', 'DarkSkin'])

    # Remove dyads with zero or negative exposure (no matches)
    df = df[df['refNum'] > 0]

    # Create offset (log of exposure)
    df['offset'] = np.log(df['refNum'])

    # Create descriptive rate variable
    df['red_rate'] = df['redCards'] / df['refNum']

    # Ensure required final columns exist; if missing in source, create as NA placeholders
    required_final_cols = [
        'DarkSkin', 'SkinTone', 'redCards', 'refNum', 'offset',
        'leagueCountry', 'club', 'player', 'goals'
    ]
    for col in required_final_cols:
        if col not in df.columns:
            df[col] = pd.NA

    # Coerce control variables to numeric where appropriate (these are conceptual numeric controls)
    # If coercion fails, those rows will be dropped below to avoid NaNs/infs in modeling matrix.
    for c in ['leagueCountry', 'club', 'player']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Ensure there are no missing values in required final columns
    df = df.dropna(subset=required_final_cols)

    # Ensure DarkSkin is integer 0/1
    df['DarkSkin'] = df['DarkSkin'].astype(int)

    # Keep only the columns necessary for modeling and diagnostics (plus rater columns and SkinCategory)
    keep_cols = [
        'rater1', 'rater2', 'SkinTone', 'SkinCategory', 'DarkSkin',
        'redCards', 'refNum', 'offset', 'red_rate',
        'leagueCountry', 'club', 'player', 'goals'
    ]
    # Guarantee order and presence
    keep_cols = [c for c in keep_cols]  # keep as is since we've ensured they exist

    return df[keep_cols]


def model(df: pd.DataFrame) -> Any:
    """
    Fit a negative binomial generalized linear model for red-card counts with exposure offset.

    Primary model specification:
      redCards ~ DarkSkin + leagueCountry + club + player
    Family: Negative Binomial (to allow for overdispersion relative to Poisson)
    Offset: log(refNum) (column 'offset') to model rate per match.

    Standard errors are clustered by referee id ('goals') to account for dependence of dyads officiated by the same referee.

    Returns:
      results_robust: statsmodels results object with cluster-robust covariance.
    """

    # Required columns for modeling
    required = ['redCards', 'offset', 'DarkSkin', 'goals']
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Column {c} is required for the model but not found in dataframe")

    # Build design matrix (include controls if available)
    X_cols = ['DarkSkin']
    for ctrl in ['leagueCountry', 'club', 'player']:
        if ctrl in df.columns:
            X_cols.append(ctrl)

    # Prepare X, replace infinities and drop rows with missing values in predictors/response/offset/groups
    X = df[X_cols].copy()
    X = sm.add_constant(X, has_constant='add')
    X = X.replace([np.inf, -np.inf], np.nan)

    y = df['redCards']
    offset = df['offset']
    groups = df['goals']

    # Mask rows with complete data
    mask = X.notnull().all(axis=1) & y.notnull() & offset.notnull() & groups.notnull()
    if mask.sum() == 0:
        raise ValueError("No observations available for modeling after dropping missing or infinite values.")

    X = X.loc[mask]
    y = y.loc[mask]
    offset = offset.loc[mask]
    groups = groups.loc[mask]

    # Fit Negative Binomial GLM with offset
    model_nb = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset)
    result = model_nb.fit()

    # Cluster-robust standard errors by referee id (goals)
    try:
        results_robust = result.get_robustcov_results(cov_type='cluster', groups=groups)
    except Exception:
        import warnings
        warnings.warn('Cluster-robust covariance estimation failed; returning unclustered fit')
        results_robust = result

    print(results_robust.summary())
    return results_robust