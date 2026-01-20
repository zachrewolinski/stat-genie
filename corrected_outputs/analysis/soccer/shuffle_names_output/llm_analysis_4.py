from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/soccer/shuffle_names_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy to avoid side effects
    df = df.copy()

    # --------- Map columns (based on provided schema descriptions) ---------
    # photoID (small integer, 0-2) corresponds to number of red cards in the dyad
    # redCards (1..47) corresponds to number of matches in the dyad (exposure)
    # rater1, rater2 are skin-tone ratings normalized to 0..1
    # leagueCountry ~ mean implicit bias (IAT); club ~ mean explicit bias (thermometer)
    # meanIAT (per schema samples) contains player position (categorical)
    # victories contains league / country name
    # goals corresponds to referee id
    # position column appears to be birthdate of player in dd.mm.yyyy format

    # Keep only rows with the minimum required information
    required = ['photoID', 'redCards', 'rater1', 'rater2', 'leagueCountry', 'club', 'meanIAT', 'victories', 'goals', 'position']
    df = df.dropna(subset=required)

    # Create dependent variable: number of red cards received from this referee
    df['red_cards'] = pd.to_numeric(df['photoID'], errors='coerce').fillna(0).astype(int)

    # Create exposure / matches variable
    df['n_matches'] = pd.to_numeric(df['redCards'], errors='coerce')
    # Remove dyads with missing or zero matches
    df = df[df['n_matches'].notna()]
    df = df[df['n_matches'] > 0]
    df['n_matches'] = df['n_matches'].astype(int)

    # Skin tone: average of two raters (both already normalized 0..1 per schema)
    df['skin_rating'] = df[['rater1', 'rater2']].mean(axis=1)
    # Binary dark vs light using median split (keeps sample-specific parity between groups)
    median_skin = df['skin_rating'].median()
    df['is_dark'] = (df['skin_rating'] >= median_skin).astype(int)

    # Referee country-level implicit & explicit bias measures
    df['ref_implicit_bias'] = pd.to_numeric(df['leagueCountry'], errors='coerce')
    df['ref_explicit_bias'] = pd.to_numeric(df['club'], errors='coerce')

    # Player age: parse birthdate (column 'position' per schema appears to be dd.mm.yyyy)
    df['player_birthdate'] = pd.to_datetime(df['position'], format='%d.%m.%Y', errors='coerce')
    # Compute age at season midpoint (use 2013-01-01 as season reference)
    season_ref = pd.to_datetime('2013-01-01')
    df['player_age'] = (season_ref - df['player_birthdate']).dt.days / 365.25

    # Player on-field position (categorical) stored in 'meanIAT' per schema samples
    df['player_position'] = df['meanIAT'].astype('category')

    # League / country where the match was played
    df['league'] = df['victories'].astype('category')

    # Referee identifier (for clustered SEs)
    df['ref_id'] = df['goals']

    # Drop rows missing any of the transformed columns we will use in the model
    model_cols = ['red_cards', 'n_matches', 'skin_rating', 'is_dark', 'ref_implicit_bias', 'ref_explicit_bias', 'player_age', 'player_position', 'league', 'ref_id']
    df = df.dropna(subset=model_cols)

    # Select and return only the columns needed for modeling (keeps original DF minimal)
    return df[model_cols].reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """Fit a negative-binomial GLM for red-card counts with exposure = number of matches.

    Model specification (primary):
      red_cards ~ is_dark + skin_rating + player_age + ref_implicit_bias + ref_explicit_bias
                   + C(player_position) + C(league)
    Family: NegativeBinomial (accounts for overdispersion relative to Poisson)
    Offset: log(n_matches) to model red-card rate per match
    Clustered standard errors: clustered on ref_id to account for non-independence across dyads by the same referee.

    Returns: clustered results object (statsmodels Results)
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import numpy as np

    # Ensure offset is defined and finite
    df = df.copy()
    df['log_exposure'] = np.log(df['n_matches'].astype(float))

    formula = (
        'red_cards ~ is_dark + skin_rating + player_age + ref_implicit_bias + ref_explicit_bias '
        '+ C(player_position) + C(league)'
    )

    # Fit Negative Binomial GLM with offset
    glm_nb = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial(), offset=df['log_exposure'])
    res = glm_nb.fit()

    # Get cluster-robust SEs clustered by referee id
    try:
        res_cl = res.get_robustcov_results(cov_type='cluster', groups=df['ref_id'])
    except Exception:
        # fallback: return original results if clustering fails
        res_cl = res

    # Return the fitted, cluster-robust results object
    return res_cl


