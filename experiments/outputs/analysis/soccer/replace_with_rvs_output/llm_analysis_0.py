from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dyad dataframe into a modeling dataframe with the following additions/cleaning:
    - Parse birthdays
    - Compute skin_tone_avg from rater1 and rater2 (require both raters present for the binary comparison)
    - Create a binary dark vs light indicator (dark_binary) using conservative thresholds (>=0.66 = Dark, <=0.33 = Light)
    - Remove intermediate skin-tone rows so the analysis compares clear Dark vs clear Light
    - Drop rows with missing redCards or games and drop rows with games <= 0 (cannot use offset log(0))
    - Compute player age (years) at 2013-01-01
    - Compute red_rate for descriptive checks
    - Keep relevant columns for modeling
    """
    df = df.copy()

    # Ensure necessary columns exist
    required = [
        'rater1', 'rater2', 'redCards', 'games', 'birthday', 'refNum', 'refCountry',
        'meanIAT', 'meanExp', 'height', 'weight', 'position', 'leagueCountry',
        'goals', 'yellowCards'
    ]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    # Drop rows missing both raters or missing redCards/games (we require both raters for the conservative binary classification)
    df = df.dropna(subset=['rater1', 'rater2', 'redCards', 'games'])

    # Convert games and redCards to numeric (coerce non-numeric to NaN)
    df['games'] = pd.to_numeric(df['games'], errors='coerce')
    df['redCards'] = pd.to_numeric(df['redCards'], errors='coerce')

    # Drop rows with missing or zero/negative games (cannot take log(0) for offset)
    df = df[df['games'].notna()]
    df = df[df['games'] > 0]

    # Compute average skin tone score (ensure raters numeric)
    df['rater1'] = pd.to_numeric(df['rater1'], errors='coerce')
    df['rater2'] = pd.to_numeric(df['rater2'], errors='coerce')
    df['skin_tone_avg'] = df[['rater1', 'rater2']].mean(axis=1)

    # Define binary Dark vs Light using conservative thresholds
    # rater scale normalized to [0,1]: 0 = very light, 1 = very dark
    df['skin_tone_group'] = pd.NA
    df.loc[df['skin_tone_avg'] <= 0.33, 'skin_tone_group'] = 'Light'
    df.loc[df['skin_tone_avg'] >= 0.66, 'skin_tone_group'] = 'Dark'

    # Keep only clear Dark vs clear Light for main comparison
    df = df[df['skin_tone_group'].isin(['Light', 'Dark'])].copy()

    # Binary indicator: 1 = Dark, 0 = Light
    df['dark_binary'] = (df['skin_tone_group'] == 'Dark').astype(int)

    # Parse birthday to compute age (some birthdays may be in 'dd.mm.yyyy' format)
    def parse_bday(x):
        # Try explicit day.month.year first, then fall back to automatic parsing
        try:
            # pd.to_datetime with errors='coerce' will return NaT instead of raising
            parsed = pd.to_datetime(x, format='%d.%m.%Y', errors='coerce')
            if pd.isna(parsed):
                parsed = pd.to_datetime(x, errors='coerce')
            return parsed
        except Exception:
            return pd.to_datetime(x, errors='coerce')

    df['birthday_parsed'] = df['birthday'].apply(parse_bday)
    # Use a fixed reference date approximately in the middle of the 2012-2013 season
    ref_date = pd.Timestamp('2013-01-01')
    df['age'] = (ref_date - df['birthday_parsed']).dt.days / 365.25

    # Compute red rate for diagnostics
    df['red_rate'] = df['redCards'] / df['games']

    # Ensure numeric controls are numeric and fill if needed for diagnostics (model will handle medians later)
    for col in ['meanIAT', 'meanExp', 'height', 'weight', 'goals', 'yellowCards']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Keep only columns needed for modeling (but return others useful for diagnostics)
    keep_cols = [
        'playerShort', 'player', 'club', 'leagueCountry', 'birthday_parsed', 'age', 'height', 'weight',
        'position', 'games', 'redCards', 'red_rate', 'yellowCards', 'yellowReds', 'goals', 'photoID',
        'rater1', 'rater2', 'skin_tone_avg', 'skin_tone_group', 'dark_binary',
        'refNum', 'refCountry', 'meanIAT', 'nIAT', 'seIAT', 'meanExp', 'nExp', 'seExp'
    ]

    # Keep only columns present in the dataframe
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit a count model to test whether dark-skinned players receive more red cards than light-skinned players.

    Primary model: Negative Binomial regression of redCards on dark_binary (1=Dark, 0=Light),
    with log(games) as an offset (exposure), and controlling for country-level bias measures and player covariates.
    Cluster standard errors by referee (refNum).

    Returns the fitted result object with clustered robust standard errors if clustering worked,
    otherwise returns a results object with a heteroskedasticity-robust covariance type.
    """
    df = df.copy()

    # Ensure required columns exist and drop rows with missing values in core model columns
    required_model_cols = ['redCards', 'games', 'dark_binary', 'meanIAT', 'meanExp', 'age',
                           'height', 'weight', 'goals', 'yellowCards', 'position', 'leagueCountry', 'refNum']
    for col in ['redCards', 'games', 'dark_binary']:
        if col not in df.columns:
            raise ValueError(f"Required column missing from input dataframe: {col}")

    df = df.dropna(subset=['redCards', 'games', 'dark_binary'])

    # Fill missing controls with column medians where appropriate (keep strict NA handling for key covariates)
    for col in ['meanIAT', 'meanExp', 'age', 'height', 'weight', 'goals', 'yellowCards']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # fill with median of available values
            if df[col].notna().any():
                df[col] = df[col].fillna(df[col].median())
            else:
                # if entire column is NA, fill with 0 to avoid errors (rare)
                df[col] = df[col].fillna(0)

    # Convert categorical variables to category dtype for formula handling
    if 'position' in df.columns:
        df['position'] = df['position'].astype('category')
    if 'leagueCountry' in df.columns:
        df['leagueCountry'] = df['leagueCountry'].astype('category')

    # Build formula: main predictor dark_binary, controls, and categorical fixed effects for position and leagueCountry
    formula = (
        'redCards ~ dark_binary + meanIAT + meanExp + age + height + weight + goals + yellowCards'
        ' + C(position) + C(leagueCountry)'
    )

    # Offset is log(games) to model rate per game
    offset = np.log(df['games'].astype(float))

    # Fit Negative Binomial GLM, attempting to request clustered robust SEs at fit time.
    # If clustering fails (e.g., unsupported statsmodels version), fall back to HC3 robust cov.
    try:
        # Many statsmodels versions accept cov_type and cov_kwds in the fit() call for GLMResults
        result = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial(), offset=offset).fit(
            cov_type='cluster', cov_kwds={'groups': df['refNum']}
        )
    except Exception:
        # Fallback: fit with HC3 robust covariance
        result = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial(), offset=offset).fit(
            cov_type='HC3'
        )

    return result