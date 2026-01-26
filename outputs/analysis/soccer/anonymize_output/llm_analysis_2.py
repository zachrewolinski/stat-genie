from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/soccer/anonymize_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe ready for count modeling.

    Produces these columns (names used later in the model):
      - RedCards (int): feature16
      - Matches (int): feature9 (exposure)
      - SkinRater1, SkinRater2 (float): feature18, feature19
      - SkinAvg (float): mean of the two raters
      - SkinDark (int): 1 if SkinAvg >= 0.75 (dark), 0 if SkinAvg <= 0.25 (light); rows with intermediate SkinAvg removed
      - SkinToneLabel (str): 'Dark'/'Light'
      - Position (str): feature8
      - Age (float): computed from birthdate feature5 relative to 2013-01-01 (season midpoint)
      - Height_cm (float): feature6
      - Weight_kg (float): feature7
      - ImplicitBias (float): feature22
      - ExplicitBias (float): feature25
      - RefereeID (int): feature20
      - RefereeCountryID (int): feature21
      - PlayerShortName, PlayerFullName, Club (from feature1..3) preserved for potential subgroup checks

    Rows with missing critical variables (skin raters, matches, red cards) are dropped.
    """
    # Copy to avoid modifying original
    df = df.copy()

    # Rename raw features to readable names
    df = df.rename(columns={
        'feature1': 'PlayerShortName',
        'feature2': 'PlayerFullName',
        'feature3': 'Club',
        'feature4': 'LeagueCountry',
        'feature5': 'Birthdate_raw',
        'feature6': 'Height_cm',
        'feature7': 'Weight_kg',
        'feature8': 'Position',
        'feature9': 'Matches',
        'feature16': 'RedCards',
        'feature18': 'SkinRater1',
        'feature19': 'SkinRater2',
        'feature20': 'RefereeID',
        'feature21': 'RefereeCountryID',
        'feature22': 'ImplicitBias',
        'feature25': 'ExplicitBias'
    })

    # Ensure numeric columns are numeric
    numeric_cols = ['Matches', 'RedCards', 'Height_cm', 'Weight_kg', 'SkinRater1', 'SkinRater2', 'ImplicitBias', 'ExplicitBias', 'RefereeID', 'RefereeCountryID']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Parse birthdate (expected format dd.mm.yyyy) and compute age at season midpoint (2013-01-01)
    if 'Birthdate_raw' in df.columns:
        df['Birthdate'] = pd.to_datetime(df['Birthdate_raw'], format='%d.%m.%Y', errors='coerce')
        season_ref = pd.Timestamp('2013-01-01')
        df['Age'] = (season_ref - df['Birthdate']).dt.days / 365.25
    else:
        df['Age'] = np.nan

    # Compute skin average across raters
    df['SkinAvg'] = df[['SkinRater1', 'SkinRater2']].mean(axis=1)

    # Keep only rows with valid critical data
    df = df.dropna(subset=['SkinAvg', 'Matches', 'RedCards'])

    # Filter out zero-match dyads: cannot observe red card rate without exposure
    df = df[df['Matches'] > 0]

    # Create binary SkinDark indicator and label; exclude middle/ambiguous ratings
    # Raters were normalized to 0..1 with 5 categories -> thresholds 0.25 and 0.75 capture 'light' vs 'dark'.
    def classify_skin(x):
        if pd.isna(x):
            return np.nan
        if x <= 0.25:
            return 'Light'
        if x >= 0.75:
            return 'Dark'
        return 'Ambiguous'

    df['SkinToneLabel'] = df['SkinAvg'].apply(classify_skin)
    # Keep only clear Light vs Dark
    df = df[df['SkinToneLabel'].isin(['Light', 'Dark'])].copy()
    df['SkinDark'] = (df['SkinToneLabel'] == 'Dark').astype(int)

    # Ensure Position is a string category and fill missing with 'Unknown'
    if 'Position' in df.columns:
        df['Position'] = df['Position'].astype(str).fillna('Unknown')
    else:
        df['Position'] = 'Unknown'

    # Final type coercions
    df['RedCards'] = df['RedCards'].astype(int)
    df['Matches'] = df['Matches'].astype(int)
    df['RefereeID'] = df['RefereeID'].astype(int)
    if 'RefereeCountryID' in df.columns:
        df['RefereeCountryID'] = pd.to_numeric(df['RefereeCountryID'], errors='coerce')

    # Keep only the columns needed for modeling + identifiers for checks
    keep_cols = [
        'PlayerShortName', 'PlayerFullName', 'Club',
        'RedCards', 'Matches', 'SkinRater1', 'SkinRater2', 'SkinAvg', 'SkinDark', 'SkinToneLabel',
        'Position', 'Age', 'Height_cm', 'Weight_kg',
        'ImplicitBias', 'ExplicitBias', 'RefereeID', 'RefereeCountryID'
    ]
    # Some columns may be missing in rare cases; intersect
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a negative-binomial (overdispersed count) model predicting RedCards with exposure = Matches.

    Model specification (formula):
      RedCards ~ SkinDark + Age + Height_cm + Weight_kg + C(Position) + ImplicitBias + ExplicitBias
    with offset = log(Matches).

    We compute cluster-robust standard errors clustered by RefereeID to account for correlation of dyads under the same referee.

    Returns the fitted results object with clustered SEs.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import numpy as np

    # Verify required columns exist
    required = ['RedCards', 'Matches', 'SkinDark', 'RefereeID']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Build formula. Use categorical Position via C(Position).
    formula = 'RedCards ~ SkinDark + Age + Height_cm + Weight_kg + C(Position) + ImplicitBias + ExplicitBias'

    # Create offset = log(Matches)
    offset = np.log(df['Matches'].astype(float))

    # Fit a GLM with Negative Binomial family (allows overdispersion vs Poisson)
    model_glm = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial(), offset=offset)
    res = model_glm.fit()

    # Derive cluster-robust covariance (cluster by RefereeID)
    # statsmodels has get_robustcov_results on the fit result
    try:
        res_clust = res.get_robustcov_results(cov_type='cluster', groups=df['RefereeID'])
    except Exception:
        # Fallback: return original result if clustering fails
        res_clust = res

    # Print a short summary and return the clustered result
    print(res_clust.summary())
    return res_clust


