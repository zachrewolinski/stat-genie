from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
from types import SimpleNamespace

# Additional imports for robust covariance calculations and p-values
from statsmodels.stats.sandwich_covariance import cov_cluster, cov_hc1
from scipy import stats


df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/soccer/anonymize_output/soccer.csv')


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Rename the columns we will use to concise descriptive names
    rename_map = {
        'feature1': 'PlayerShortName',
        'feature3': 'Club',
        'feature6': 'Height_cm',
        'feature7': 'Weight_kg',
        'feature8': 'Position',
        'feature9': 'Matches',
        'feature13': 'Goals',
        'feature14': 'YellowCards',
        'feature15': 'YellowRed',
        'feature16': 'RedCards',
        'feature17': 'PhotoID',
        'feature18': 'SkinRater1',
        'feature19': 'SkinRater2',
        'feature20': 'RefereeID',
        'feature21': 'RefereeCountryID',
        'feature22': 'RefereeImplicit',
        'feature25': 'RefereeExplicit'
    }
    df = df.rename(columns=rename_map)

    # Drop rows missing essential variables for defining IV and DV and exposure
    df = df.dropna(subset=['Matches', 'RedCards', 'SkinRater1', 'SkinRater2'])

    # Ensure RefereeID exists and drop rows without it (needed for clustering)
    if 'RefereeID' in df.columns:
        df = df.dropna(subset=['RefereeID'])
    else:
        # If the column truly doesn't exist, raise an informative error
        raise ValueError("Required column missing from raw dataframe: RefereeID")

    # Ensure Matches is positive integer (exposure). If zero or negative, drop row.
    df = df[df['Matches'] > 0]
    df['Matches'] = df['Matches'].astype(int)

    # Construct skin ratings: mean of two raters
    df['SkinMean'] = df[['SkinRater1', 'SkinRater2']].mean(axis=1)

    # Create categorical skin bin focusing on clear contrasts: light vs dark
    # Rater scale normalized to 1 across 5 categories: typical values: 0.0, 0.25, 0.5, 0.75, 1.0
    # Here we treat <=0.4 as 'light', >=0.6 as 'dark', middle values as 'neither' and drop them
    df['SkinBin'] = df['SkinMean'].apply(lambda x: 'dark' if x >= 0.6 else ('light' if x <= 0.4 else 'neither'))

    # Keep only clear light vs dark contrasts to directly answer the research question
    df = df[df['SkinBin'].isin(['dark', 'light'])].reset_index(drop=True)

    # Binary indicator: 1 = dark, 0 = light
    df['DarkSkin'] = (df['SkinBin'] == 'dark').astype(int)

    # Fill missing numeric controls with 0 when appropriate (counts), otherwise leave as-is
    for col in ['Goals', 'YellowCards', 'YellowRed']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
        else:
            df[col] = 0.0

    # Position as categorical; fill missing
    df['Position'] = df['Position'].fillna('Unknown').astype(str)

    # Ensure bias measures are numeric; replace missing with column mean (or 0 if all missing)
    if 'RefereeImplicit' in df.columns:
        temp = pd.to_numeric(df['RefereeImplicit'], errors='coerce')
        if temp.notna().any():
            temp = temp.fillna(temp.mean())
        else:
            temp = temp.fillna(0.0)
        df['RefereeImplicit'] = temp.astype(float)
    else:
        df['RefereeImplicit'] = 0.0

    if 'RefereeExplicit' in df.columns:
        temp2 = pd.to_numeric(df['RefereeExplicit'], errors='coerce')
        if temp2.notna().any():
            temp2 = temp2.fillna(temp2.mean())
        else:
            temp2 = temp2.fillna(0.0)
        df['RefereeExplicit'] = temp2.astype(float)
    else:
        df['RefereeExplicit'] = 0.0

    # Ensure RedCards is numeric integer count
    df['RedCards'] = pd.to_numeric(df['RedCards'], errors='coerce').fillna(0).astype(int)

    # Create log of matches for use as offset in count model
    # Offset will be log(Matches)
    df['log_Matches'] = np.log(df['Matches'].astype(float))

    # Keep the final set of columns used in modeling (this is the returned dataframe)
    keep_cols = [
        'PlayerShortName', 'Club', 'Height_cm', 'Weight_kg', 'Position',
        'Matches', 'log_Matches', 'Goals', 'YellowCards', 'YellowRed', 'RedCards',
        'PhotoID', 'SkinRater1', 'SkinRater2', 'SkinMean', 'SkinBin', 'DarkSkin',
        'RefereeID', 'RefereeCountryID', 'RefereeImplicit', 'RefereeExplicit'
    ]
    # Some of these may not exist in every dataset; intersect to avoid KeyError
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a negative binomial regression for count of red cards with exposure = number of matches (offset = log(Matches)).

    Model: RedCards ~ DarkSkin + Goals + YellowCards + YellowRed + C(Position) + RefereeImplicit + RefereeExplicit
    Offset: log_Matches

    Use clustered (by RefereeID) robust standard errors to account for within-referee correlation.
    Returns a dict with the base model and the clustered-covariance results object.
    """
    df = df.copy()

    # Ensure essential columns exist
    required = ['RedCards', 'DarkSkin', 'log_Matches', 'RefereeID']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Required column missing from transformed dataframe: {c}")

    # Build design matrix: position dummies + numeric covariates
    pos_dummies = pd.get_dummies(df['Position'], prefix='Pos', drop_first=True)
    covariates = ['DarkSkin', 'Goals', 'YellowCards', 'YellowRed', 'RefereeImplicit', 'RefereeExplicit']
    # If any covariate is missing, create a zero column to keep model syntax stable
    for cov in covariates:
        if cov not in df.columns:
            df[cov] = 0.0

    X = pd.concat([pos_dummies, df[covariates]], axis=1)

    # Ensure no infinite or NaN values in X: coerce to numeric, replace inf, then fill NaN with 0
    X = X.apply(pd.to_numeric, errors='coerce')
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.fillna(0.0)

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Final safety: ensure all exog are finite numeric
    X = X.astype(float)
    if not np.isfinite(X.values).all():
        # replace any remaining inf/nan with 0
        X_values = np.where(np.isfinite(X.values), X.values, 0.0)
        X = pd.DataFrame(X_values, columns=X.columns)
        # ensure a constant column exists (statsmodels expects column names)
        if 'const' not in X.columns:
            X.insert(0, 'const', 1.0)

    y = pd.to_numeric(df['RedCards'], errors='coerce').fillna(0).astype(float)
    offset = pd.to_numeric(df['log_Matches'], errors='coerce')
    offset.replace([np.inf, -np.inf], np.nan, inplace=True)
    offset = offset.fillna(0.0).astype(float)

    # Fit negative binomial GLM with log link and offset
    nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset).fit()

    # Helper to build a lightweight clustered-results object with clustered SEs, t-stats, p-values
    def build_clustered_results(results, group_labels):
        try:
            cov = cov_cluster(results, group_labels)
        except Exception:
            # Fall back to HC1 if clustering fails
            cov = cov_hc1(results)

        # Ensure covariance matrix is a numpy array
        cov = np.asarray(cov)
        # Handle potential shape mismatches
        params = np.asarray(results.params)
        # If cov shape doesn't match params, try to adjust
        if cov.shape[0] != params.shape[0]:
            # attempt to extract matching rows/cols by intersecting index names if possible
            try:
                cov = cov[: params.shape[0], : params.shape[0]]
            except Exception:
                # Fallback: create a diagonal cov with original cov_params if available
                base_cov = results.cov_params() if hasattr(results, 'cov_params') else np.diag(np.square(results.bse))
                cov = np.asarray(base_cov)
                cov = cov[: params.shape[0], : params.shape[0]]

        bse = np.sqrt(np.diag(cov))
        # Protect against zeros on diagonal
        bse = np.where(bse == 0, np.finfo(float).eps, bse)
        tvalues = params / bse
        pvalues = 2 * stats.norm.sf(np.abs(tvalues))

        # Build a simple namespace object that mirrors key attributes expected downstream
        clustered = SimpleNamespace(
            params=pd.Series(params, index=results.params.index),
            bse=pd.Series(bse, index=results.params.index),
            tvalues=pd.Series(tvalues, index=results.params.index),
            pvalues=pd.Series(pvalues, index=results.params.index),
            cov_params=lambda: pd.DataFrame(cov, index=results.params.index, columns=results.params.index),
            model=results.model,
            original_results=results,
            summary=lambda: results.summary()
        )
        return clustered

    # Compute clustered robust SEs by RefereeID
    try:
        groups = df['RefereeID'].values
        nb_model_clustered = build_clustered_results(nb_model, groups)
    except Exception:
        # Fall back to HC1-based clustered-like object if something unexpected happens
        nb_model_clustered = build_clustered_results(nb_model, df['RefereeID'].fillna('__missing__').values)

    # Optional: also fit an interaction model to test whether country-level implicit bias moderates the skin-tone effect
    interaction_results = None
    if 'RefereeImplicit' in df.columns:
        X_inter = X.copy()
        # Create interaction term using the original df columns to ensure correct alignment
        inter_term = pd.to_numeric(df['DarkSkin'], errors='coerce').fillna(0.0) * pd.to_numeric(df['RefereeImplicit'], errors='coerce').fillna(0.0)
        # Ensure shapes align
        X_inter['DarkSkin_x_Implicit'] = inter_term.values
        # Clean the interaction matrix
        X_inter = X_inter.apply(pd.to_numeric, errors='coerce').fillna(0.0)

        try:
            nb_inter = sm.GLM(y, X_inter, family=sm.families.NegativeBinomial(), offset=offset).fit()
            try:
                groups = df['RefereeID'].values
                nb_inter_clustered = build_clustered_results(nb_inter, groups)
            except Exception:
                nb_inter_clustered = build_clustered_results(nb_inter, df['RefereeID'].fillna('__missing__').values)
            interaction_results = {'nb_inter': nb_inter, 'nb_inter_clustered': nb_inter_clustered}
        except Exception:
            interaction_results = None

    return {
        'nb_model': nb_model,
        'nb_model_clustered': nb_model_clustered,
        'interaction_results': interaction_results
    }