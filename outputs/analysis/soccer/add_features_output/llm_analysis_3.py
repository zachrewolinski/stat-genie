from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/soccer/add_features_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into an analysis-ready dataframe for modeling the effect of skin tone (dark vs light) on red card counts.

    Output dataframe columns used in the model (exact names):
      - redCards
      - games
      - offset_log_games
      - DarkSkin (binary: 1 = dark, 0 = light)
      - age_c, height_c, weight_c, yellowCards_c, goals_c
      - meanIAT_c, meanExp_c
      - pos_* (one-hot dummies for position)
      - league_* (one-hot dummies for leagueCountry)
      - refNum
    """
    df = df.copy()

    # Required columns: rater1, rater2, games, redCards, refNum
    # Drop rows missing key variables
    req = ['rater1', 'rater2', 'games', 'redCards', 'refNum']
    df = df.dropna(subset=req)

    # Ensure games is positive integer (exposure). Drop rows with games <= 0
    df = df[df['games'] > 0]

    # Compute mean skin rating across the two independent raters
    df['SkinAvg'] = df[['rater1', 'rater2']].mean(axis=1)

    # Define extremes: light vs dark.
    def skin_cat(x):
        if x <= 0.25:
            return 'Light'
        elif x >= 0.75:
            return 'Dark'
        else:
            return 'Intermediate'

    df['SkinCat'] = df['SkinAvg'].apply(skin_cat)

    # Keep only Light and Dark to answer the research question (contrast)
    df = df[df['SkinCat'].isin(['Light', 'Dark'])].copy()

    # Binary variable: DarkSkin = 1 if Dark, 0 if Light
    df['DarkSkin'] = (df['SkinCat'] == 'Dark').astype(int)

    # Offset: log of games (exposure)
    df['offset_log_games'] = np.log(df['games'].astype(float))

    # Center continuous controls to improve interpretability
    cont_cols = ['age', 'height', 'weight', 'yellowCards', 'goals', 'meanIAT', 'meanExp']
    for col in cont_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col + '_c'] = df[col] - df[col].mean()
        else:
            # create the required final column even if original missing
            df[col + '_c'] = np.nan

    # One-hot encode categorical controls: position and leagueCountry
    if 'position' in df.columns:
        pos_dummies = pd.get_dummies(df['position'].fillna('Unknown'), prefix='pos')
        # drop first to avoid multicollinearity
        if pos_dummies.shape[1] > 1:
            pos_dummies = pos_dummies.iloc[:, 1:]
        df = pd.concat([df, pos_dummies], axis=1)

    if 'leagueCountry' in df.columns:
        league_dummies = pd.get_dummies(df['leagueCountry'].fillna('Unknown'), prefix='league')
        if league_dummies.shape[1] > 1:
            league_dummies = league_dummies.iloc[:, 1:]
        df = pd.concat([df, league_dummies], axis=1)

    # Collect position and league dummy names dynamically
    pos_cols = [c for c in df.columns if c.startswith('pos_')]
    league_cols = [c for c in df.columns if c.startswith('league_')]

    # Ensure redCards is numeric
    df['redCards'] = pd.to_numeric(df['redCards'], errors='coerce')

    # Keep only columns needed for modeling + refNum
    keep_cols = (
        ['redCards', 'games', 'offset_log_games', 'DarkSkin', 'refNum']
        + ['age_c', 'height_c', 'weight_c', 'yellowCards_c', 'goals_c', 'meanIAT_c', 'meanExp_c']
        + pos_cols + league_cols
    )

    # Some of the keep_cols might not exist (pos/league dummies), so intersect with df.columns
    keep_cols = [c for c in keep_cols if c in df.columns]

    df = df[keep_cols].copy()

    # Final drop of any rows with missing outcome, DarkSkin, offset, or numeric controls
    required_for_model = ['redCards', 'DarkSkin', 'offset_log_games',
                          'age_c', 'height_c', 'weight_c', 'yellowCards_c', 'goals_c', 'meanIAT_c', 'meanExp_c']
    required_for_model = [c for c in required_for_model if c in df.columns]
    # Drop rows with NA in any of these required numeric/model vars
    if required_for_model:
        df = df.dropna(subset=required_for_model)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a negative binomial regression for red card counts with exposure offset = log(games).
    The main coefficient of interest is DarkSkin. We include controls and cluster standard
    errors at the referee level (refNum).

    Returns the fitted model results object and a small summary DataFrame for the DarkSkin effect
    including incidence rate ratio (IRR) and 95% CI.
    """
    df = df.copy()

    # Identify regressors automatically
    base_regs = ['DarkSkin', 'age_c', 'height_c', 'weight_c', 'yellowCards_c', 'goals_c', 'meanIAT_c', 'meanExp_c']
    reg_cols = [c for c in base_regs if c in df.columns]
    # append any pos_ and league_ dummies
    reg_cols += [c for c in df.columns if c.startswith('pos_') or c.startswith('league_')]

    # Prepare X, y, and offset
    X = df[reg_cols].copy().astype(float) if reg_cols else pd.DataFrame(index=df.index)
    # Replace infinities with NaN to allow clean dropping
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    y = pd.to_numeric(df['redCards'], errors='coerce').astype(float)
    offset = pd.to_numeric(df['offset_log_games'], errors='coerce').astype(float)

    # Ensure refNum present for clustering
    if 'refNum' not in df.columns:
        raise ValueError("refNum column is required in the final dataframe for clustering.")
    ref = df['refNum']

    # Drop any rows with missing data in X, y, offset, or refNum
    mask = pd.Series(True, index=df.index)
    if not X.empty:
        mask &= X.notnull().all(axis=1)
    mask &= y.notnull()
    mask &= offset.notnull()
    mask &= ref.notnull()

    if mask.sum() == 0:
        raise ValueError("No usable rows after dropping missing data for model fitting.")

    X_clean = X.loc[mask].copy()
    y_clean = y.loc[mask].copy()
    offset_clean = offset.loc[mask].copy()
    ref_clean = ref.loc[mask].copy()

    # Add constant
    if X_clean.shape[1] > 0:
        X_clean = sm.add_constant(X_clean, has_constant='add')
    else:
        # If there are no regressors (should not happen because DarkSkin is required), create only constant
        X_clean = sm.add_constant(pd.DataFrame(index=X_clean.index), has_constant='add')

    # Fit Negative Binomial GLM with offset; cluster standard errors by refNum
    try:
        model_nb = sm.GLM(y_clean, X_clean, family=sm.families.NegativeBinomial(), offset=offset_clean)
        res_nb = model_nb.fit()
        clustered = res_nb.get_robustcov_results(cov_type='cluster', groups=ref_clean)
        results = clustered
    except Exception:
        # fallback to non-clustered results if clustering fails
        model_nb = sm.GLM(y_clean, X_clean, family=sm.families.NegativeBinomial(), offset=offset_clean)
        results = model_nb.fit()

    # Prepare a small summary for DarkSkin: coefficient, IRR and 95% CI
    if 'DarkSkin' in X_clean.columns and 'DarkSkin' in getattr(results, 'params', {}).index:
        params = results.params
        bse = results.bse
        coef = params.get('DarkSkin', np.nan)
        se = bse.get('DarkSkin', np.nan)
        z = coef / se if (not np.isnan(coef) and not np.isnan(se) and se != 0) else np.nan
        ci_lower = coef - 1.96 * se if (not np.isnan(coef) and not np.isnan(se)) else np.nan
        ci_upper = coef + 1.96 * se if (not np.isnan(coef) and not np.isnan(se)) else np.nan
        irr = np.exp(coef) if not np.isnan(coef) else np.nan
        irr_lower = np.exp(ci_lower) if not np.isnan(ci_lower) else np.nan
        irr_upper = np.exp(ci_upper) if not np.isnan(ci_upper) else np.nan

        dark_summary = pd.DataFrame({
            'term': ['DarkSkin'],
            'coef': [coef],
            'se': [se],
            'z': [z],
            'ci_lower': [ci_lower],
            'ci_upper': [ci_upper],
            'IRR': [irr],
            'IRR_95ci_low': [irr_lower],
            'IRR_95ci_high': [irr_upper]
        })
    else:
        dark_summary = pd.DataFrame()

    return {"model_result": results, "dark_skin_summary": dark_summary}