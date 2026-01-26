from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/soccer/add_features_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw player-referee dyad dataframe into an analysis-ready dataframe.

    Steps performed:
    - Ensure numeric types for relevant columns and drop rows missing required fields
    - Compute mean rater score and derive a categorical SkinTone ('Dark', 'Light', 'Other')
    - Filter to players classified as 'Dark' or 'Light' (research question compares these groups)
    - Create a log of games (log_games) to use as an offset in count models
    - Fill or coerce commonly-missing control variables in a principled way
    - Return a reduced dataframe containing only columns required for modelling
    """
    df = df.copy()

    # Ensure numeric coercion for key columns
    df['redCards'] = pd.to_numeric(df['redCards'], errors='coerce')
    df['games'] = pd.to_numeric(df['games'], errors='coerce')
    df['rater1'] = pd.to_numeric(df.get('rater1'), errors='coerce')
    df['rater2'] = pd.to_numeric(df.get('rater2'), errors='coerce')

    # Required columns must be present for the analysis; drop rows missing essential info
    required = ['redCards', 'games', 'rater1', 'rater2', 'position', 'leagueCountry', 'refNum']
    df = df.dropna(subset=required)

    # Remove dyads with zero games (no exposure) as we cannot model counts with zero exposure
    df = df[df['games'] > 0]

    # Mean rater score (rater1 and rater2 are on a normalized 0-1 scale representing 5-point ratings)
    df['mean_rater'] = df[['rater1', 'rater2']].mean(axis=1)

    # Create categorical skin tone: keep only clear 'Light' vs 'Dark' contrast
    # Original 5-point normalized scale likely takes values {0.0,0.25,0.5,0.75,1.0}
    df['SkinTone'] = df['mean_rater'].apply(lambda x: 'Dark' if x >= 0.75 else ('Light' if x <= 0.25 else 'Other'))

    # Filter to only 'Dark' and 'Light' players to directly test the research question
    df = df[df['SkinTone'].isin(['Dark', 'Light'])].reset_index(drop=True)

    # Offset: log of games (exposure)
    df['log_games'] = np.log(df['games'])

    # Convert and impute other numeric controls
    df['goals'] = pd.to_numeric(df.get('goals'), errors='coerce').fillna(0)
    df['yellowCards'] = pd.to_numeric(df.get('yellowCards'), errors='coerce').fillna(0)
    df['age'] = pd.to_numeric(df.get('age'), errors='coerce')
    # If age is missing, fill with median age (reasonable for control variable)
    if df['age'].isnull().any():
        df['age'] = df['age'].fillna(df['age'].median())

    # country-level bias measures: if missing, fill with overall mean (keeps observations while avoiding dropping rows)
    df['meanIAT'] = pd.to_numeric(df.get('meanIAT'), errors='coerce')
    df['meanExp'] = pd.to_numeric(df.get('meanExp'), errors='coerce')
    if df['meanIAT'].isnull().any():
        df['meanIAT'] = df['meanIAT'].fillna(df['meanIAT'].mean())
    if df['meanExp'].isnull().any():
        df['meanExp'] = df['meanExp'].fillna(df['meanExp'].mean())

    # Keep only columns necessary for modeling and downstream checks
    keep_cols = [
        'playerShort', 'refNum', 'leagueCountry', 'position',
        'games', 'log_games', 'redCards', 'mean_rater', 'SkinTone',
        'goals', 'yellowCards', 'age', 'meanIAT', 'meanExp'
    ]

    # Some columns may not exist in the raw df; guard by intersection
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # Ensure categorical fields are of type object/string for formula based modeling
    if 'position' in df.columns:
        df['position'] = df['position'].astype(str)
    if 'leagueCountry' in df.columns:
        df['leagueCountry'] = df['leagueCountry'].astype(str)
    if 'playerShort' in df.columns:
        df['playerShort'] = df['playerShort'].astype(str)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a negative binomial regression of redCards on SkinTone (Dark vs Light) with exposure offset = log(games).

    Model specification (fixed-effects):
      redCards ~ C(SkinTone) + goals + yellowCards + age + meanIAT + meanExp + C(position) + C(leagueCountry)
    Family: Negative Binomial (to account for overdispersion commonly present in count data)
    Offset: log_games (log of games in dyad)

    Standard errors: cluster-robust by referee (refNum) to account for within-referee correlation across dyads.

    Returns the fitted results object with clustered robust covariance (if possible).
    """
    import statsmodels.formula.api as smf

    # Ensure the offset column exists
    if 'log_games' not in df.columns:
        df['log_games'] = np.log(df['games'].replace(0, np.nan)).fillna(0.0)

    formula = (
        'redCards ~ C(SkinTone) + goals + yellowCards + age + meanIAT + meanExp'
        ' + C(position) + C(leagueCountry)'
    )

    # Fit negative binomial GLM with offset
    nb_model = smf.glm(formula=formula, data=df,
                       family=sm.families.NegativeBinomial(),
                       offset=df['log_games'])

    fit = nb_model.fit()

    # Convert to clustered robust covariance by referee if possible
    # Use try/except because get_robustcov_results may fail on some older statsmodels versions
    try:
        results = fit.get_robustcov_results(cov_type='cluster', groups=df['refNum'])
    except Exception:
        # Fall back to the plain fit if robust covariance cannot be computed
        results = fit

    # Print summary for quick inspection (caller may still inspect returned object)
    try:
        print(results.summary())
    except Exception:
        pass

    return results


