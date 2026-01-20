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
    Transform raw dataset into analysis-ready dataframe with the following columns (names used in modeling):
      - RedCards (feature16)
      - Matches (feature9)
      - SkinScore (mean(feature18, feature19))
      - SkinDark (binary: 1=Dark extreme, 0=Light extreme) — rows with mid ratings removed
      - Position (feature8)
      - LeagueCountry (feature4)
      - RefImplicit (feature22)
      - RefExplicit (feature25)
      - Height_cm (feature6)
      - Weight_kg (feature7)
      - Age (computed from feature5 at 2012-09-01)
      - Goals (feature13)
      - YellowCards (feature14)
      - RefID (feature20)
      - PlayerID (feature1)

    Steps:
      - rename columns for clarity
      - compute SkinScore and restrict to Dark vs Light extremes
      - compute Age
      - drop rows with missing critical values and with Matches == 0
    """
    import numpy as np
    import pandas as pd

    # Make a copy to avoid modifying input
    df = df.copy()

    # Rename relevant features to meaningful names (map based on provided schema)
    rename_map = {
        'feature1': 'PlayerID',
        'feature4': 'LeagueCountry',
        'feature5': 'Birthdate',
        'feature6': 'Height_cm',
        'feature7': 'Weight_kg',
        'feature8': 'Position',
        'feature9': 'Matches',
        'feature13': 'Goals',
        'feature14': 'YellowCards',
        'feature16': 'RedCards',
        'feature18': 'Rater1_Skin',
        'feature19': 'Rater2_Skin',
        'feature20': 'RefID',
        'feature21': 'RefCountryID',
        'feature22': 'RefImplicit',
        'feature25': 'RefExplicit'
    }
    df = df.rename(columns=rename_map)

    # Ensure numeric columns are numeric
    for col in ['Rater1_Skin', 'Rater2_Skin', 'Matches', 'RedCards', 'Height_cm', 'Weight_kg', 'Goals', 'YellowCards', 'RefImplicit', 'RefExplicit']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Compute SkinScore as the mean of two independent raters
    df['SkinScore'] = df[['Rater1_Skin', 'Rater2_Skin']].mean(axis=1)

    # Define extremes: original raters used a 5-point scale normalized to 1 (values likely 0,0.25,0.5,0.75,1)
    # We'll treat the bottom two categories as 'Light' and the top two as 'Dark', dropping the middle category
    # thresholds chosen to capture the two extreme groups: <=0.375 -> Light, >=0.625 -> Dark
    def _skin_cat(x):
        if pd.isna(x):
            return np.nan
        if x <= 0.375:
            return 'Light'
        if x >= 0.625:
            return 'Dark'
        return 'Mid'

    df['SkinCategory'] = df['SkinScore'].apply(_skin_cat)

    # Keep only the extremes (Light and Dark) for the primary comparison
    df = df[df['SkinCategory'].isin(['Light', 'Dark'])]

    # Binary indicator: 1 = Dark, 0 = Light
    df['SkinDark'] = (df['SkinCategory'] == 'Dark').astype(int)

    # Parse birthdate and compute age at a reference date in the 2012-2013 season (use 2012-09-01)
    # Birthdate format given as dd.mm.yyyy
    if 'Birthdate' in df.columns:
        df['Birthdate_parsed'] = pd.to_datetime(df['Birthdate'], format='%d.%m.%Y', errors='coerce')
        ref_date = pd.to_datetime('2012-09-01')
        df['Age'] = (ref_date - df['Birthdate_parsed']).dt.days / 365.25
    else:
        df['Age'] = np.nan

    # Keep relevant columns and drop rows with missing critical analysis values
    keep_cols = [
        'PlayerID', 'RefID', 'RefCountryID', 'LeagueCountry', 'Position',
        'Matches', 'RedCards', 'SkinScore', 'SkinDark', 'SkinCategory',
        'Height_cm', 'Weight_kg', 'Age', 'Goals', 'YellowCards',
        'RefImplicit', 'RefExplicit'
    ]

    # Some columns may be missing in input; keep only those present
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    # Drop rows with missing DV, IV, or exposure
    df = df.dropna(subset=['RedCards', 'SkinScore', 'Matches', 'RefID'])

    # Ensure integer counts and positive matches
    df['RedCards'] = df['RedCards'].astype(int)
    df['Matches'] = pd.to_numeric(df['Matches'], errors='coerce')
    df = df.dropna(subset=['Matches'])
    # Remove dyads with zero recorded matches (can't be used as exposure)
    df = df[df['Matches'] > 0]

    # Reset index
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a negative binomial regression for the count of red cards (RedCards) with Matches as exposure.
    Primary independent variable: SkinDark (1=Dark, 0=Light). We also include SkinScore as a continuous sensitivity check.
    Controls: Position (categorical), LeagueCountry (categorical), Height_cm, Weight_kg, Age, Goals, YellowCards,
              RefImplicit, RefExplicit.

    We cluster standard errors at the referee level (RefID) to account for referee-specific tendencies.

    Returns the model results object with cluster-robust standard errors.
    """
    import numpy as np
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Build formula: RedCards ~ SkinDark + SkinScore + controls + categorical dummies for Position and LeagueCountry
    # Using GLM with a Negative Binomial family and offset = log(Matches) (exposure)
    formula_parts = [
        'SkinDark',
        'SkinScore',
        'Height_cm',
        'Weight_kg',
        'Age',
        'Goals',
        'YellowCards',
        'RefImplicit',
        'RefExplicit',
        'C(Position)',
        'C(LeagueCountry)'
    ]
    # Keep only terms that are present in the dataframe
    available_terms = [t for t in formula_parts if (t.split('(')[-1].strip(')') in df.columns) or (t in df.columns) or (t.startswith('C(') and t[2:-1] in df.columns)]
    formula = 'RedCards ~ ' + ' + '.join(available_terms)

    # Fit the model
    # Use the Matches column as an exposure (offset = log(Matches))
    offset = np.log(df['Matches'].astype(float))

    # Use GLM with Negative Binomial family
    model = smf.glm(formula, data=df, family=sm.families.NegativeBinomial(), offset=offset)
    res = model.fit()

    # Obtain cluster-robust covariance (clustered by referee ID)
    # If RefID is present, use it for clustering; otherwise, return standard results
    if 'RefID' in df.columns:
        try:
            res_cluster = res.get_robustcov_results(cov_type='cluster', groups=df['RefID'])
        except Exception:
            # If cluster robust fails for any reason, fall back to default
            res_cluster = res
    else:
        res_cluster = res

    # Return the results object with cluster-robust cov if available
    return res_cluster


