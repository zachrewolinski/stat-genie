from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/soccer/noperturb_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset into analysis-ready dataframe.
    Produces:
      - SkinAvg: average of rater1 and rater2 (continuous 0..1)
      - SkinCategory: 'Light', 'Medium', or 'Dark' (based on SkinAvg)
      - IsDark: binary indicator (1 = Dark, 0 = Light). Rows with 'Medium' are dropped to compare extremes.
      - age: age in years at reference date 2013-01-01 computed from 'birthday'
    Filters out rows with missing critical fields and ensures games >= 1 (required for offset).
    """
    df = df.copy()

    # Basic required columns exist check is left to upstream code; assume columns from schema are present.
    # Drop rows missing primary coder ratings, outcome, or games
    df = df.dropna(subset=['rater1', 'rater2', 'redCards', 'games'])

    # Compute average skin rating
    df['SkinAvg'] = (df['rater1'] + df['rater2']) / 2.0

    # Categorize extremes: Light <= 0.4, Dark >= 0.6, Medium otherwise
    df['SkinCategory'] = df['SkinAvg'].apply(lambda x: 'Light' if x <= 0.4 else ('Dark' if x >= 0.6 else 'Medium'))

    # Keep only extremes to answer the contrast 'dark' vs 'light'
    df = df[df['SkinCategory'].isin(['Light', 'Dark'])]

    # Binary indicator for dark skin
    df['IsDark'] = (df['SkinCategory'] == 'Dark').astype(int)

    # Parse birthday and compute age at reference date (season midpoint)
    # birthdays are dd.mm.yyyy per schema
    df['birthday'] = pd.to_datetime(df['birthday'], format='%d.%m.%Y', errors='coerce')
    ref_date = pd.to_datetime('2013-01-01')
    df['age'] = (ref_date - df['birthday']).dt.days / 365.25

    # Ensure games is numeric and >= 1 to allow log offset; drop games < 1
    df['games'] = pd.to_numeric(df['games'], errors='coerce')
    df = df[df['games'].notna() & (df['games'] >= 1)]

    # Cast redCards to integer counts
    df['redCards'] = pd.to_numeric(df['redCards'], errors='coerce').fillna(0).astype(int)

    # Drop rows with missing values in the control variables used in the model
    control_cols = ['age', 'height', 'weight', 'goals', 'yellowCards', 'meanIAT', 'meanExp', 'position', 'refNum']
    df = df.dropna(subset=control_cols)

    # Keep columns that will be used in modeling to keep the dataframe compact
    keep_cols = ['playerShort', 'player', 'club', 'leagueCountry', 'birthday', 'height', 'weight', 'position',
                 'games', 'victories', 'ties', 'defeats', 'goals', 'yellowCards', 'yellowReds', 'redCards',
                 'photoID', 'rater1', 'rater2', 'SkinAvg', 'SkinCategory', 'IsDark', 'refNum', 'refCountry',
                 'meanIAT', 'nIAT', 'seIAT', 'meanExp', 'nExp', 'seExp', 'age']

    # Keep only columns that exist in the dataframe (schema may vary)
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a negative binomial regression of redCards (count) on IsDark (main IV) with an offset for games
    and controls for player-level and country-level covariates. Cluster standard errors by referee (refNum).

    Model formula:
      redCards ~ IsDark + age + height + weight + goals + yellowCards + meanIAT + meanExp + C(position)
    Offset: log(games)

    Returns the fitted model results with clustered robust standard errors by refNum.
    """
    import statsmodels.formula.api as smf

    # Ensure we are working with a copy
    df = df.copy()

    # Build formula. Position is included as categorical factor.
    formula = 'redCards ~ IsDark + age + height + weight + goals + yellowCards + meanIAT + meanExp + C(position)'

    # Create offset = log(games)
    offset = np.log(df['games'].astype(float))

    # Fit GLM with Negative Binomial family (accounts for overdispersion relative to Poisson)
    model_glm = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial(), offset=offset)
    result = model_glm.fit()

    # Obtain clustered robust standard errors clustered on referee id (refNum)
    # Use get_robustcov_results to produce a results object with clustered covariances
    try:
        result_clustered = result.get_robustcov_results(cov_type='cluster', groups=df['refNum'])
    except Exception:
        # If clustering fails for some reason, return the original result but warn the user
        print('Warning: clustering by refNum failed; returning unclustered results.')
        result_clustered = result

    # Print and return clustered results
    print(result_clustered.summary())
    return result_clustered


