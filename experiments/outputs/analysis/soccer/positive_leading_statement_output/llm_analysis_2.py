from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/soccer/positive_leading_statement_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform original dataset into the analytic dataframe used for modeling.

    Produced columns (exact names used by the model):
      - redCards: count outcome (numeric)
      - games: number of games in dyad (numeric, used as offset)
      - AvgSkin: mean of rater1 and rater2 (continuous 0-1)
      - SkinDark: binary indicator (1 = dark skin (AvgSkin >= 0.75), 0 = light skin (AvgSkin <= 0.25)). Rows with intermediate AvgSkin are removed.
      - age: player age in years (numeric)
      - height, weight: numeric controls
      - position: categorical control (kept as-is for formula-based categorical handling)
      - leagueCountry: categorical control
      - meanIAT, meanExp: country-level bias measures (numeric)
      - refNum, refCountry: referee identifiers used for clustering / descriptive checks
    """
    df = df.copy()

    # Ensure required columns exist and drop rows with missing core values
    required_core = ['rater1', 'rater2', 'redCards', 'games']
    df = df.dropna(subset=required_core)

    # Compute average skin rating
    df['AvgSkin'] = (pd.to_numeric(df['rater1'], errors='coerce') + pd.to_numeric(df['rater2'], errors='coerce')) / 2.0

    # Define clear light and clear dark thresholds and create binary SkinDark
    # Normalized scale assumed (0.0..1.0) with five discrete values; thresholds chosen to select clear light and clear dark
    df['SkinDark'] = np.where(df['AvgSkin'] >= 0.75, 1,
                               np.where(df['AvgSkin'] <= 0.25, 0, np.nan))

    # Keep only rows that are clearly light or clearly dark to address measurement noise and create a clean contrast
    df = df[df['SkinDark'].notna()].copy()
    df['SkinDark'] = df['SkinDark'].astype(int)

    # Parse birthday and compute age at a reference date (season midpoint ~ 2013-01-01)
    # birthday format documented as 'dd.mm.yyyy'
    df['birthday'] = pd.to_datetime(df['birthday'], format='%d.%m.%Y', errors='coerce')
    reference_date = pd.to_datetime('2013-01-01')
    df['age'] = ((reference_date - df['birthday']).dt.days / 365.25).astype(float)

    # Ensure numeric types for physical covariates and outcome/exposure
    df['height'] = pd.to_numeric(df['height'], errors='coerce')
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
    df['games'] = pd.to_numeric(df['games'], errors='coerce')
    df['redCards'] = pd.to_numeric(df['redCards'], errors='coerce')

    # Ensure bias measures are numeric
    df['meanIAT'] = pd.to_numeric(df['meanIAT'], errors='coerce')
    df['meanExp'] = pd.to_numeric(df['meanExp'], errors='coerce')

    # Keep only rows with the covariates needed for the model
    keep_cols = ['redCards', 'games', 'AvgSkin', 'SkinDark', 'age', 'height', 'weight',
                 'position', 'leagueCountry', 'meanIAT', 'meanExp', 'refNum', 'refCountry']

    df = df.dropna(subset=keep_cols)

    # Ensure games > 0 for offset (dataset minimum is 1 according to schema, but double-check)
    df = df[df['games'] > 0].copy()

    # Return a dataframe with exactly the columns the model code expects
    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a negative binomial GLM for red card counts with an offset for number of games.

    Primary model (tests research question):
      redCards ~ SkinDark + age + height + weight + C(position) + C(leagueCountry) + meanIAT + meanExp
      offset = log(games)

    Standard errors are clustered by referee (refNum) to account for correlations across dyads judged by the same referee.

    Returns:
      - fitted results object from statsmodels (GLMResults)
    """
    import statsmodels.formula.api as smf

    # Copy dataframe to avoid side-effects
    df = df.copy()

    # Formula: main independent variable is SkinDark (1 dark, 0 light). Position and leagueCountry treated as categorical.
    formula = 'redCards ~ SkinDark + age + height + weight + C(position) + C(leagueCountry) + meanIAT + meanExp'

    # Build and fit the negative binomial GLM with log(games) as an offset (exposure)
    # Note: statsmodels expects the offset to be provided as a numeric array; we give log(games)
    model = smf.glm(formula=formula,
                    data=df,
                    family=sm.families.NegativeBinomial(),
                    offset=np.log(df['games']))

    # Fit with cluster-robust SEs by refNum (referee)
    results = model.fit(cov_type='cluster', cov_kwds={'groups': df['refNum']})

    # Print summary for quick inspection; return the results object for programmatic use
    print(results.summary())
    return results


