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
    Transform the raw dataset into a modeling dataframe. Returns a dataframe with the exact columns used by the model.

    Steps:
    - Compute average skin rating across two independent raters (feature18, feature19)
    - Define extreme skin groups: 'Light' (avg <= 0.25) and 'Dark' (avg >= 0.75). Keep only these extremes to compare dark vs light.
    - Parse birthdate to compute Age (reference date set to 2013-06-01 as the season reference).
    - Rename/derive modeling columns: RedCards, Matches, Goals, YellowCards, Position, RefereeID, RefereeCountryID, RefCountryImplicitBias, RefCountryExplicitBias.
    - Drop rows with missing values for required modeling columns.
    """
    df = df.copy()

    # Compute average skin rating
    df['SkinRatingAvg'] = df[['feature18', 'feature19']].mean(axis=1)

    # Define extreme groups to compare (clear Light vs clear Dark). Thresholds chosen to select the endpoints of the 5-point normalized scale.
    df['SkinGroup'] = pd.cut(df['SkinRatingAvg'], bins=[-0.1, 0.25, 0.75, 1.1], labels=['Light', 'Mid', 'Dark'])
    # Keep only the clear extremes (Light, Dark)
    df = df[df['SkinGroup'].isin(['Light', 'Dark'])]

    # Map group to binary indicator: IsDark = 1 for Dark, 0 for Light
    df['IsDark'] = (df['SkinGroup'] == 'Dark').astype(int)

    # Dependent variable and exposure
    df['RedCards'] = pd.to_numeric(df['feature16'], errors='coerce').astype('Int64')
    df['Matches'] = pd.to_numeric(df['feature9'], errors='coerce').astype('Int64')

    # Controls: goals, yellow cards
    df['Goals'] = pd.to_numeric(df['feature13'], errors='coerce').astype('Int64')
    df['YellowCards'] = pd.to_numeric(df['feature14'], errors='coerce').astype('Int64')

    # Referee identifiers and country-level bias measures
    df['RefereeID'] = pd.to_numeric(df['feature20'], errors='coerce').astype('Int64')
    df['RefereeCountryID'] = pd.to_numeric(df['feature21'], errors='coerce').astype('Int64')
    df['RefCountryImplicitBias'] = pd.to_numeric(df['feature22'], errors='coerce')
    df['RefCountryExplicitBias'] = pd.to_numeric(df['feature25'], errors='coerce')

    # Position
    df['Position'] = df['feature8'].astype(str)

    # Compute Age in years using a season reference date (approx mid-2013)
    df['Birthdate'] = pd.to_datetime(df['feature5'], format='%d.%m.%Y', errors='coerce')
    reference_date = pd.to_datetime('2013-06-01')
    df['Age'] = (reference_date - df['Birthdate']).dt.days / 365.25

    # Drop rows missing any variables necessary for the main model
    required = ['SkinRatingAvg', 'SkinGroup', 'IsDark', 'RedCards', 'Matches', 'Goals', 'YellowCards',
                'RefereeID', 'RefereeCountryID', 'RefCountryImplicitBias', 'RefCountryExplicitBias', 'Age', 'Position']

    df = df.dropna(subset=required)

    # Cast numeric Int64 columns back to plain integers where safe
    df['RedCards'] = df['RedCards'].astype(int)
    df['Matches'] = df['Matches'].astype(int)
    df['Goals'] = df['Goals'].astype(int)
    df['YellowCards'] = df['YellowCards'].astype(int)
    df['RefereeID'] = df['RefereeID'].astype(int)
    df['RefereeCountryID'] = df['RefereeCountryID'].astype(int)

    # Select and return only the columns needed for modeling and diagnostics
    keep_cols = [
        'IsDark',
        'SkinRatingAvg',
        'SkinGroup',
        'RedCards',
        'Matches',
        'Age',
        'Position',
        'Goals',
        'YellowCards',
        'RefereeID',
        'RefereeCountryID',
        'RefCountryImplicitBias',
        'RefCountryExplicitBias'
    ]

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a negative binomial regression predicting the count of red cards (RedCards) with exposure equal to number of Matches.

    Model specification (primary):
      RedCards ~ IsDark + Age + Goals + YellowCards + C(Position) + RefCountryImplicitBias + RefCountryExplicitBias
      offset = log(Matches)

    We cluster standard errors at the referee level (RefereeID) to account for multiple dyads judged by the same referee.

    Returns the fitted results object (statsmodels result) so the caller can inspect coefficients, summaries, and diagnostics.
    """
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    df = df.copy()

    # Prepare design matrix: numeric covariates and position dummies (drop_first to avoid multicollinearity)
    cat_pos = pd.get_dummies(df['Position'], prefix='Pos', drop_first=True)

    X_numeric = df[['IsDark', 'Age', 'Goals', 'YellowCards', 'RefCountryImplicitBias', 'RefCountryExplicitBias']]
    X = pd.concat([X_numeric, cat_pos], axis=1)
    X = sm.add_constant(X, has_constant='add')

    y = df['RedCards']

    # Offset is log of matches (exposure). Ensure Matches > 0 (dataset min is 1 but guard anyway)
    offset = np.log(df['Matches'].replace(0, 1))

    # Fit negative binomial GLM with clustered SE by RefereeID
    model = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset)

    # Fit and request cluster-robust covariance by referee ID
    try:
        res = model.fit(cov_type='cluster', cov_kwds={'groups': df['RefereeID']})
    except Exception:
        # Fallback to default fit if cluster robust fails for some reason
        res = model.fit()

    return res


