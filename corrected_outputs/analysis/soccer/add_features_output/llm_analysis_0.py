from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/soccer/add_features_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Keep rows where games is positive and redCards is non-missing
    df = df[df['games'].notna()]
    df = df[df['games'] > 0]
    df = df[df['redCards'].notna()]

    # Build the outcome as explicit column
    df['red_cards'] = df['redCards'].astype(int)

    # Compute average skin tone from the two raters (skipna: if one is missing use the other)
    df['SkinToneAvg'] = df[['rater1', 'rater2']].mean(axis=1)

    # Drop rows without any skin-tone rating
    df = df[~df['SkinToneAvg'].isna()]

    # Create a binary DarkSkin indicator using a median split of SkinToneAvg
    # 1 = darker-than-median (darker), 0 = lighter-or-equal-than-median (lighter)
    median_skin = df['SkinToneAvg'].median()
    df['DarkSkin'] = (df['SkinToneAvg'] > median_skin).astype(int)

    # Exposure offset: log of games (will be used in the model)
    # (we keep raw games column as specified in the conceptual variables)
    df['log_games'] = np.log(df['games'].astype(float))

    # Interaction term: DarkSkin x meanIAT (tests whether referee implicit bias moderates effect)
    # Fill missing meanIAT with the country's median before creating interaction to avoid NaNs
    if 'meanIAT' in df.columns:
        df['meanIAT'] = df['meanIAT'].astype(float)
        df['meanIAT'] = df['meanIAT'].fillna(df['meanIAT'].median())
    else:
        df['meanIAT'] = 0.0
    df['DarkSkin_meanIAT'] = df['DarkSkin'] * df['meanIAT']

    # Ensure meanExp exists and fill missing with median
    if 'meanExp' in df.columns:
        df['meanExp'] = df['meanExp'].astype(float)
        df['meanExp'] = df['meanExp'].fillna(df['meanExp'].median())
    else:
        df['meanExp'] = 0.0

    # Numeric controls: fill missing with column medians (conservative imputation)
    num_ctrls = ['age', 'height', 'weight', 'goals', 'yellowCards']
    for c in num_ctrls:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            df[c] = df[c].fillna(df[c].median())
        else:
            # create column of zeros if missing
            df[c] = 0

    # Position dummies (drop first to avoid multicollinearity)
    if 'position' in df.columns:
        pos_dummies = pd.get_dummies(df['position'].fillna('Missing'), prefix='pos', drop_first=True)
        df = pd.concat([df, pos_dummies], axis=1)
    else:
        # no position column: ensure no pos dummies exist
        pass

    # Referee country dummies (drop first); refCountry may be numeric id - cast to string to create stable dummies
    if 'refCountry' in df.columns:
        refcountry_dummies = pd.get_dummies(df['refCountry'].astype(str).fillna('Missing'), prefix='refCountry', drop_first=True)
        df = pd.concat([df, refcountry_dummies], axis=1)
    else:
        pass

    # Ensure refNum exists (used for clustering)
    if 'refNum' not in df.columns:
        df['refNum'] = 0

    # Final sanity: keep only rows with finite log_games
    df = df[np.isfinite(df['log_games'])]

    # Reset index for a clean dataframe
    df = df.reset_index(drop=True)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # Prepare dependent and independent variables for count regression with exposure offset
    df = df.copy()

    # Dependent variable
    y = df['red_cards'].astype(float)

    # Base covariates (these names must appear in transformed df)
    base_vars = [
        'DarkSkin',           # main IV
        'meanIAT',            # moderator / control
        'DarkSkin_meanIAT',   # interaction term
        'meanExp',
        'age',
        'height',
        'weight',
        'goals',
        'yellowCards'
    ]

    # Add position dummies if present (columns that start with 'pos_')
    pos_cols = [c for c in df.columns if c.startswith('pos_')]
    refcountry_cols = [c for c in df.columns if c.startswith('refCountry_')]

    # Final design matrix columns
    X_cols = base_vars + pos_cols + refcountry_cols

    # Keep only columns that exist (safe-guard)
    X_cols = [c for c in X_cols if c in df.columns]

    X = df[X_cols].astype(float)
    X = sm.add_constant(X, has_constant='add')

    # Offset = log(games) stored in column log_games
    offset = df['log_games'].astype(float)

    # Fit Negative Binomial GLM (preferred for overdispersed counts). Cluster standard errors by referee (refNum).
    # If NegativeBinomial is not appropriate, Poisson with robust SE could be used as a fallback.
    try:
        model_glm = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset)
    except Exception:
        # fallback to Poisson
        model_glm = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)

    # Fit with clustered standard errors by refNum
    results = model_glm.fit(cov_type='cluster', cov_kwds={'groups': df['refNum']})

    # Return the fitted results object (contains params, summary(), etc.)
    return results


