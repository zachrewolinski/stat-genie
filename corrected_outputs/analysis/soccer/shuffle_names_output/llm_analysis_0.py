from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/soccer/shuffle_names_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset to create the analysis dataframe.

    Produces the following columns used in modeling:
      - avg_skin: mean of rater1 and rater2 (intermediate diagnostic column)
      - SkinDark: binary indicator (1 = dark-skinned, 0 = light-skinned) defined by top/bottom terciles of avg_skin
      - red_card_count: number of red cards received from the referee in the dyad (assumed to be stored in 'photoID')
      - matches: number of matches in the player-referee dyad (assumed to be stored in 'redCards')
      - meanIAT_z, club_z, playerShort_z: z-scored control variables
      - goals: referee id (kept for clustering SEs)
    """
    df = df.copy()

    # Ensure numeric columns are numeric (coerce non-numeric -> NaN)
    for col in ['rater1', 'rater2', 'photoID', 'redCards', 'meanIAT', 'club', 'playerShort', 'goals']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing key fields required to build the outcome/exposure and skin measures
    df = df.dropna(subset=['rater1', 'rater2', 'photoID', 'redCards'])

    # Average the two rater scores to form a continuous skin tone measure
    df['avg_skin'] = (df['rater1'] + df['rater2']) / 2.0

    # Define light vs dark by bottom vs top terciles of avg_skin
    q_low = df['avg_skin'].quantile(0.33)
    q_high = df['avg_skin'].quantile(0.67)

    df = df[(df['avg_skin'] <= q_low) | (df['avg_skin'] >= q_high)].copy()
    df['SkinDark'] = (df['avg_skin'] >= q_high).astype(int)  # 1 = dark (top tercile), 0 = light (bottom tercile)

    # Build dependent variables
    df['red_card_count'] = pd.to_numeric(df['photoID'], errors='coerce').astype(float)
    df['matches'] = pd.to_numeric(df['redCards'], errors='coerce').astype(float)

    # Remove dyads with zero or missing exposure to avoid log(0) in offset
    df = df[df['matches'] > 0].copy()

    # Standardize continuous controls (z-scores). If a control is missing entirely, fill with 0s.
    for raw, zname in [('meanIAT', 'meanIAT_z'), ('club', 'club_z'), ('playerShort', 'playerShort_z')]:
        if raw in df.columns and df[raw].notnull().any():
            mean = df[raw].mean(skipna=True)
            std = df[raw].std(ddof=0, skipna=True)
            if std == 0 or np.isnan(std):
                df[zname] = 0.0
            else:
                df[zname] = (df[raw] - mean) / std
            # Replace any remaining NaNs (rows where the raw value was NaN) with 0.0 (average)
            df[zname] = df[zname].fillna(0.0)
        else:
            df[zname] = 0.0

    # Keep only the columns required for modeling (plus avg_skin for diagnostics)
    keep_cols = ['SkinDark', 'red_card_count', 'matches', 'avg_skin', 'meanIAT_z', 'club_z', 'playerShort_z', 'goals']
    keep_present = [c for c in keep_cols if c in df.columns]

    return df[keep_present]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a count regression to test whether dark-skinned players are more likely
    to receive red cards than light-skinned players.

    Returns a fitted results object (cluster-robust if possible).
    """
    df = df.copy()

    # Ensure required columns exist
    required = ['red_card_count', 'matches', 'SkinDark']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column for modeling: {col}")

    # Prepare design matrix
    X_cols = []
    if 'SkinDark' in df.columns:
        X_cols.append('SkinDark')
    for ctrl in ['meanIAT_z', 'club_z', 'playerShort_z']:
        if ctrl in df.columns:
            X_cols.append(ctrl)

    X = df[X_cols].copy()
    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Ensure X is numeric and has no infinite values; replace inf with NaN then fill NaN with 0 (mean)
    X = X.apply(pd.to_numeric, errors='coerce')
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0.0, inplace=True)
    # Align y and offset
    y = pd.to_numeric(df['red_card_count'], errors='coerce')
    offset = np.log(pd.to_numeric(df['matches'], errors='coerce').astype(float))

    # Drop rows with missing outcome or offset
    valid_idx = y.notnull() & offset.notnull()
    if not valid_idx.all():
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        offset = offset.loc[valid_idx]
        df = df.loc[valid_idx]

    # Final safety: ensure no NaNs/Infs remain in X, y, offset
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0.0, inplace=True)
    if not np.isfinite(y.values).all():
        raise ValueError("Non-finite values found in outcome after cleaning.")
    if not np.isfinite(offset.values).all():
        raise ValueError("Non-finite values found in offset after cleaning.")

    # Fit Negative Binomial GLM with offset, fall back to Poisson on failure
    try:
        model_glm = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset)
        res = model_glm.fit()
    except Exception:
        model_glm = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
        res = model_glm.fit()

    # Attempt to compute cluster-robust SEs clustered by referee id (goals) if present
    if 'goals' in df.columns:
        try:
            groups = df['goals'].loc[X.index]
            res_clustered = res.get_robustcov_results(cov_type='cluster', groups=groups)
            return res_clustered
        except Exception:
            return res
    else:
        return res