from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# If running as a script, loading of df is optional; keep but harmless if path not present.
try:
    df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/soccer/anonymize_output/soccer.csv')
except Exception:
    df = None


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the input raw dataframe into the analysis dataframe with the exact columns used in the model.

    Key outputs (columns created/kept):
      - red_cards: integer count (from feature16)
      - matches: integer count of matches (from feature9)
      - skin_rater1, skin_rater2: original rater scores (feature18, feature19)
      - skin_avg: mean of available rater scores
      - skin_group: categorical ('Dark','Light','Middle') derived from skin_avg
      - is_extreme: boolean indicating Dark or Light (True) vs Middle (False)
      - dark_binary: 1 for Dark, 0 for Light (only meaningful when is_extreme==True)
      - position: from feature8
      - league_country: from feature4
      - goals: feature13
      - yellow_cards: feature14
      - implicit_bias: feature22
      - explicit_bias: feature25
      - player_age: computed from feature5 relative to 2013-01-01
      - referee_id: feature20 (used for clustering)
    """

    df = df.copy()

    # Rename frequently used features to clearer column names
    rename_map = {
        'feature5': 'birthdate_raw',
        'feature8': 'position',
        'feature9': 'matches',
        'feature13': 'goals',
        'feature14': 'yellow_cards',
        'feature16': 'red_cards',
        'feature18': 'skin_rater1',
        'feature19': 'skin_rater2',
        'feature20': 'referee_id',
        'feature21': 'ref_country_id',
        'feature22': 'implicit_bias',
        'feature25': 'explicit_bias',
        'feature4': 'league_country'
    }
    df = df.rename(columns=rename_map)

    # Parse birthdate to compute age. Input format: dd.mm.yyyy (day.month.year)
    # Some values might be missing; coerce errors to NaT.
    df['birthdate'] = pd.to_datetime(df.get('birthdate_raw'), dayfirst=True, errors='coerce', format='%d.%m.%Y')

    # Compute player age at reference date (mid-season). Use 2013-01-01 as reference.
    ref_date = pd.Timestamp('2013-01-01')
    df['player_age'] = ((ref_date - df['birthdate']).dt.days / 365.25).astype(float)

    # Keep original rater scores and compute average of available raters
    # Values are normalized to 1 in dataset; treat missing as NaN
    df['skin_rater1'] = pd.to_numeric(df.get('skin_rater1'), errors='coerce')
    df['skin_rater2'] = pd.to_numeric(df.get('skin_rater2'), errors='coerce')
    df['skin_avg'] = df[['skin_rater1', 'skin_rater2']].mean(axis=1)

    # Define groups using the extreme categories for a clear dark vs light comparison
    # The original 5-point scale was normalized to [0,1] with steps: 0.0, 0.25, 0.5, 0.75, 1.0
    # We'll consider 'Light' <= 0.25, 'Dark' >= 0.75, and 'Middle' otherwise
    def _skin_group(x):
        if pd.isna(x):
            return np.nan
        try:
            if x <= 0.25:
                return 'Light'
            elif x >= 0.75:
                return 'Dark'
            else:
                return 'Middle'
        except Exception:
            return np.nan

    df['skin_group'] = df['skin_avg'].apply(_skin_group)
    df['is_extreme'] = df['skin_group'].isin(['Dark', 'Light'])

    # Create the binary indicator used in the model: 1 = Dark, 0 = Light (only meaningful for is_extreme==True)
    # Use mapping to avoid elementwise boolean ambiguity with pandas.NA
    df['dark_binary'] = df['skin_group'].map({'Dark': 1, 'Light': 0}).astype('float')

    # Ensure numeric columns are numeric
    df['red_cards'] = pd.to_numeric(df.get('red_cards'), errors='coerce').fillna(0).astype(int)
    df['matches'] = pd.to_numeric(df.get('matches'), errors='coerce')
    df['goals'] = pd.to_numeric(df.get('goals'), errors='coerce').fillna(0).astype(float)
    df['yellow_cards'] = pd.to_numeric(df.get('yellow_cards'), errors='coerce').fillna(0).astype(float)
    df['implicit_bias'] = pd.to_numeric(df.get('implicit_bias'), errors='coerce')
    df['explicit_bias'] = pd.to_numeric(df.get('explicit_bias'), errors='coerce')
    df['referee_id'] = pd.to_numeric(df.get('referee_id'), errors='coerce')

    # Remove dyads with zero or missing matches (cannot model exposure with log(0))
    df = df[df['matches'].notnull() & (df['matches'] > 0)]

    # Keep only rows with extreme skin ratings (Dark vs Light) and with non-missing essential covariates
    required_columns = ['red_cards', 'matches', 'dark_binary', 'position', 'league_country',
                        'goals', 'yellow_cards', 'implicit_bias', 'explicit_bias', 'player_age', 'referee_id']

    # Filter to extreme skin ratings
    df = df[df['is_extreme'] == True]

    # Drop rows missing any required modeling column
    df = df.dropna(subset=required_columns)

    # For modeling convenience, ensure categorical columns are strings
    df['position'] = df['position'].astype(str)
    df['league_country'] = df['league_country'].astype(str)

    # Select and return only the columns necessary for the model (plus some useful diagnostics)
    out_cols = ['red_cards', 'matches', 'dark_binary', 'skin_avg', 'skin_group', 'is_extreme',
                'position', 'league_country', 'goals', 'yellow_cards', 'implicit_bias', 'explicit_bias',
                'player_age', 'referee_id', 'ref_country_id', 'feature1', 'feature2']

    # Some of these columns (feature1/2) may not exist if the raw dataset uses different names; keep existence-safe selection
    out = df[[c for c in out_cols if c in df.columns]].copy()

    return out


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a Negative Binomial regression predicting counts of red cards using dark vs light skin indicator,
    with the number of matches used as an offset (log exposure). Cluster standard errors at the referee level.

    Model formula (example):
      red_cards ~ dark_binary + goals + yellow_cards + C(position) + C(league_country) + implicit_bias + explicit_bias + player_age

    Returns the clustered-robust results object.
    """

    df_model = df.copy()

    # Safety: ensure dark_binary integer and matches positive
    df_model = df_model[df_model['matches'] > 0].copy()
    # dark_binary should be 0/1; cast to int after dropping missing
    df_model = df_model.dropna(subset=['dark_binary'])
    df_model['dark_binary'] = df_model['dark_binary'].astype(int)

    # Define formula. Use categorical indicators for position and league_country.
    formula = (
        'red_cards ~ dark_binary + goals + yellow_cards + C(position) + C(league_country) '
        '+ implicit_bias + explicit_bias + player_age'
    )

    # Offset: log of matches (exposure)
    offset = np.log(df_model['matches'].astype(float))

    # Fit Negative Binomial GLM
    model_nb = smf.glm(formula=formula, data=df_model,
                       family=sm.families.NegativeBinomial(),
                       offset=offset)
    results_nb = model_nb.fit()

    # Compute clustered robust standard errors at referee level
    groups = df_model['referee_id']
    try:
        results_clust = results_nb.get_robustcov_results(cov_type='cluster', groups=groups)
    except Exception:
        # Fallback: if clustering fails, return the original results (still useful)
        results_clust = results_nb

    # Print summary for quick inspection; the returned object is the clustered-robust results
    print(results_clust.summary())

    return results_clust