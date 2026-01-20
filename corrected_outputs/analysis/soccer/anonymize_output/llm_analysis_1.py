from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/soccer/anonymize_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dyad-level dataset (feature1..feature27) into analysis-ready dataframe.

    Key output columns (used in modeling):
      - RedCards: integer count of red cards in the dyad (feature16)
      - Matches: number of matches in the dyad (feature9)
      - LogMatches: natural log of Matches (used as offset)
      - SkinToneAvg: mean skin rating across two independent raters (feature18, feature19)
      - SkinToneCat: categorical tone ('Dark' / 'Light' / 'Medium')
      - SkinToneDark: binary indicator (1 if Dark, 0 if Light)
      - Goals, YellowCards, Position, Age, RefImplicitBias, RefExplicitBias, PhotoAvailable, RefID
    """

    # Work on a copy
    df = df.copy()

    # Basic renaming/derivation of analysis variables
    df['Matches'] = pd.to_numeric(df['feature9'], errors='coerce')
    df['RedCards'] = pd.to_numeric(df['feature16'], errors='coerce').fillna(0).astype(int)
    df['YellowCards'] = pd.to_numeric(df['feature14'], errors='coerce').fillna(0)
    df['YellowRedCards'] = pd.to_numeric(df['feature15'], errors='coerce').fillna(0)
    df['Goals'] = pd.to_numeric(df['feature13'], errors='coerce').fillna(0)

    # Referee identifiers and country-level bias measures
    df['RefID'] = df['feature20']
    df['RefCountryID'] = df['feature21']
    df['RefImplicitBias'] = pd.to_numeric(df['feature22'], errors='coerce')
    df['RefExplicitBias'] = pd.to_numeric(df['feature25'], errors='coerce')

    # Skin ratings from two independent raters (normalized 0..1)
    df['SkinRater1'] = pd.to_numeric(df['feature18'], errors='coerce')
    df['SkinRater2'] = pd.to_numeric(df['feature19'], errors='coerce')

    # Player position and photo availability
    df['Position'] = df['feature8']
    df['PhotoID'] = df['feature17']
    df['PhotoAvailable'] = df['PhotoID'].notnull().astype(int)

    # Birthdate -> Age (reference date 2013-01-01, middle of 2012-13 season)
    df['Birthdate'] = pd.to_datetime(df['feature5'], format='%d.%m.%Y', dayfirst=True, errors='coerce')
    ref_date = pd.to_datetime('2013-01-01')
    df['Age'] = (ref_date - df['Birthdate']).dt.days / 365.25

    # Compute average skin tone across raters (ignores NaN if one rater missing)
    df['SkinToneAvg'] = df[['SkinRater1', 'SkinRater2']].mean(axis=1)

    # Create categorical skin tone: focus on extremes for primary test
    def _skin_cat(x):
        if pd.isnull(x):
            return np.nan
        if x >= 0.75:
            return 'Dark'
        if x <= 0.25:
            return 'Light'
        return 'Medium'

    df['SkinToneCat'] = df['SkinToneAvg'].apply(_skin_cat)
    df['SkinToneDark'] = (df['SkinToneCat'] == 'Dark').astype(int)
    df['SkinToneLight'] = (df['SkinToneCat'] == 'Light').astype(int)

    # Keep only dyads with extreme ratings (Dark or Light) for the primary comparison
    df = df[df['SkinToneCat'].isin(['Dark', 'Light'])].copy()

    # Ensure positive exposure (matches > 0)
    df = df[df['Matches'] > 0].copy()

    # Drop rows with missing essential variables
    df = df.dropna(subset=['RedCards', 'SkinToneAvg', 'Matches'])

    # Offset term for modeling
    df['LogMatches'] = np.log(df['Matches'])

    # Fill missing/na for categorical position
    df['Position'] = df['Position'].fillna('Unknown')

    # Final dataframe ready for modeling
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a negative-binomial regression for red card counts with matches as exposure (offset).

    Model specification (primary):
      RedCards ~ SkinToneDark + Goals + YellowCards + C(Position) + Age + RefImplicitBias + RefExplicitBias + PhotoAvailable
    Family: NegativeBinomial (models overdispersed count data). Offset: log(Matches).
    Clustered standard errors by RefID to account for non-independence within referees.

    Returns a fitted results object with clustered robust covariance.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Ensure required columns exist
    required = ['RedCards', 'SkinToneDark', 'Goals', 'YellowCards', 'Position', 'Age', 'RefImplicitBias', 'RefExplicitBias', 'PhotoAvailable', 'LogMatches', 'RefID']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Formula (categorical Position handled via C(Position))
    formula = 'RedCards ~ SkinToneDark + Goals + YellowCards + C(Position) + Age + RefImplicitBias + RefExplicitBias + PhotoAvailable'

    # Fit GLM Negative Binomial with offset = LogMatches
    nb_model = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial(), offset=df['LogMatches'])
    nb_res = nb_model.fit()

    # Obtain cluster-robust covariance (cluster by referee ID)
    try:
        nb_res_clust = nb_res.get_robustcov_results(cov_type='cluster', groups=df['RefID'])
    except Exception:
        # Fallback: if clustering fails, return the plain fitted model
        nb_res_clust = nb_res

    # Print brief summary and return the robust result object
    print(nb_res_clust.summary())
    return nb_res_clust


