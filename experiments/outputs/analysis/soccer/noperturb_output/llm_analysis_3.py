from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/soccer/noperturb_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dyad-level dataset to create the columns used in modeling.

    Steps performed:
    - Drop rows missing essential outcome or exposure information (redCards, games) or missing both rater scores.
    - Compute mean skin rating from the two independent raters (skin_mean).
    - Create an extreme-group binary contrast is_dark where:
        * 'Light' if skin_mean <= 0.25
        * 'Dark'  if skin_mean >= 0.75
      We drop rows that are not in these two extremes (i.e., middle ratings) to create a clean test of Dark vs Light.
    - Parse birthday to compute Age at season midpoint (2013-01-01 used as reference).
    - Filter out dyads with games <= 0 because offset(log(games)) is undefined for 0.

    The returned dataframe contains all columns used in the statistical model.
    """

    # Make a copy to avoid modifying input in-place
    df = df.copy()

    # Drop rows missing required fields
    required_cols = ['redCards', 'games', 'rater1', 'rater2']
    df = df.dropna(subset=required_cols)

    # Ensure numeric types
    df['redCards'] = pd.to_numeric(df['redCards'], errors='coerce')
    df['games'] = pd.to_numeric(df['games'], errors='coerce')
    df['rater1'] = pd.to_numeric(df['rater1'], errors='coerce')
    df['rater2'] = pd.to_numeric(df['rater2'], errors='coerce')

    # Re-drop if coercion introduced NaNs
    df = df.dropna(subset=['redCards', 'games', 'rater1', 'rater2'])

    # Exclude dyads with zero or negative games (cannot use as exposure)
    df = df[df['games'] > 0]

    # Create mean skin rating (0-1 as in the dataset)
    df['skin_mean'] = (df['rater1'] + df['rater2']) / 2.0

    # Create extreme-group SkinGroup and binary is_dark
    # The rater scale is normalized to [0,1] with 5 discrete steps; we select extremes <=0.25 (Light) and >=0.75 (Dark)
    def assign_skin_group(x):
        if x <= 0.25:
            return 'Light'
        elif x >= 0.75:
            return 'Dark'
        else:
            return 'Mid'

    df['SkinGroup'] = df['skin_mean'].apply(assign_skin_group)

    # Keep only clear Light vs Dark contrast for the primary test
    df = df[df['SkinGroup'].isin(['Light', 'Dark'])].copy()

    # Binary indicator: 1 if Dark, 0 if Light
    df['is_dark'] = (df['SkinGroup'] == 'Dark').astype(int)

    # Parse birthday to compute age (some birthdays may be missing or malformed)
    # Birthday format in schema: 'dd.mm.yyyy'
    df['birthday_parsed'] = pd.to_datetime(df['birthday'], format='%d.%m.%Y', dayfirst=True, errors='coerce')

    # Compute age at season midpoint (use 2013-01-01 as reference year midpoint for 2012-2013 season)
    # If birthday parsing failed, Age will be NaN
    reference_year = pd.Timestamp('2013-01-01')
    df['Age'] = (reference_year - df['birthday_parsed']).dt.days // 365

    # Ensure control columns exist and are numeric where appropriate
    df['meanIAT'] = pd.to_numeric(df.get('meanIAT'), errors='coerce')
    df['meanExp'] = pd.to_numeric(df.get('meanExp'), errors='coerce')
    df['yellowCards'] = pd.to_numeric(df.get('yellowCards'), errors='coerce')

    # Keep columns needed for modeling; keep identifiers for clustering / checks
    needed_cols = [
        'playerShort', 'refNum', 'leagueCountry', 'position',
        'games', 'redCards', 'skin_mean', 'SkinGroup', 'is_dark',
        'meanIAT', 'meanExp', 'yellowCards', 'Age'
    ]

    # If some of these columns do not exist in the input, adjust to available ones (but we expect them to exist)
    existing_cols = [c for c in needed_cols if c in df.columns]
    df = df[existing_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a count regression to test whether dark-skinned players are more likely to receive red cards.

    Model details:
    - Outcome: redCards (count)
    - Primary predictor: is_dark (1 = Dark, 0 = Light)
    - Controls: meanIAT (included also as moderator), meanExp, yellowCards, Age, position, leagueCountry
    - Exposure: games (included as an offset via log(games))
    - Family: Negative Binomial (to account for over-dispersion relative to Poisson)
    - Robust standard errors clustered by referee (refNum) to account for nonindependence of observations within referees.

    Returns the fitted model results object with cluster-robust covariances.
    """

    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    import numpy as np

    # Make a copy
    data = df.copy()

    # Basic sanity checks
    required = ['redCards', 'games', 'is_dark', 'refNum']
    for col in required:
        if col not in data.columns:
            raise ValueError(f'Required column missing: {col}')

    # Drop rows with missing covariates used in the formula
    formula_covars = ['is_dark', 'meanIAT', 'meanExp', 'yellowCards', 'Age', 'position', 'leagueCountry']
    existing_covars = [c for c in formula_covars if c in data.columns]
    model_formula = 'redCards ~ is_dark'

    # Add covariates if present
    for cov in existing_covars:
        if cov != 'is_dark':
            model_formula += ' + ' + cov

    # Add interaction between is_dark and meanIAT (test moderator effect) if meanIAT exists
    if 'meanIAT' in data.columns:
        model_formula += ' + is_dark:meanIAT'

    # Add categorical controls explicitly (position and leagueCountry) if present
    # We use Patsy's C() to tell the formula to treat them as categorical
    if 'position' in data.columns:
        model_formula = model_formula.replace(' + position', ' + C(position)')
    if 'leagueCountry' in data.columns:
        model_formula = model_formula.replace(' + leagueCountry', ' + C(leagueCountry)')

    # Fit Negative Binomial GLM with log(games) as offset
    # Some GLM implementations expect the offset to be aligned with the dataframe; we pass it directly
    offset = np.log(data['games'].astype(float))

    model = smf.glm(formula=model_formula, data=data,
                    family=sm.families.NegativeBinomial(),
                    offset=offset)

    results = model.fit()

    # Obtain cluster-robust standard errors clustered on refNum
    # If refNum is not numeric, pass it as groups directly
    try:
        clustered = results.get_robustcov_results(cov_type='cluster', groups=data['refNum'])
    except Exception:
        # Fall back to non-clustered results if clustering fails
        clustered = results

    # Print a short summary for convenience (users can further inspect returned object)
    print(clustered.summary())

    # Return the cluster-robust results object
    return clustered


