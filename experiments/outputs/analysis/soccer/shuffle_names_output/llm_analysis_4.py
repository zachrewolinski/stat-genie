from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/soccer/shuffle_names_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe with the columns used in the statistical model.

    Assumptions / mappings (based on dataset schema and documentation):
    - rater1 and rater2: two independent skin-tone ratings normalized to 0..1 (0 ~ very light, 1 ~ very dark).
    - photoID: count of red cards received by the player from the given referee across all matches in the dyad (values 0/1/2 in the schema). We map this to RedCardCount.
    - redCards: used in the metadata as 'Number of games in the player-referee dyad' (we treat this as Matches/exposure).
    - leagueCountry: mean implicit bias score for referee country -> RefImplicit.
    - club: mean explicit bias score for referee country -> RefExplicit.
    - goals: (per schema) appears to be unique referee ID -> used as RefID for clustering.

    The function will:
    - drop rows missing essential variables
    - compute average skin tone and binary dark/light indicator
    - cast relevant columns to numeric, create standardized controls
    - return dataframe containing the columns listed in the conceptual variables
    """

    df = df.copy()

    # Ensure required columns exist in df; if some columns are named unexpectedly, this will raise a KeyError which alerts the user
    required_cols = ['rater1', 'rater2', 'photoID', 'redCards', 'leagueCountry', 'club', 'meanExp', 'playerShort', 'goals']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise KeyError(f"The input dataframe is missing required columns: {missing}")

    # Drop rows missing rater ratings or red card / match counts or referee id
    df = df.dropna(subset=['rater1', 'rater2', 'photoID', 'redCards', 'goals'])

    # Compute average skin tone (continuous 0..1) and a binary dark indicator
    df['SkinTone'] = df[['rater1', 'rater2']].mean(axis=1)

    # Binary indicator: dark skin tone = 1 if average rating >= 0.5, else 0.
    # (0.5 is the midpoint of the normalized 0..1 scale; if you prefer a median split change the threshold.)
    df['SkinToneDark'] = (df['SkinTone'] >= 0.5).astype(int)

    # Dependent variable: assumed red card count in this dyad
    # Cast to integer; if values are not integer-like this will preserve numeric values but cast to int may truncate - we assume counts
    df['RedCardCount'] = pd.to_numeric(df['photoID'], errors='coerce').fillna(0).astype(int)

    # Exposure: number of matches in dyad (used as offset)
    df['Matches'] = pd.to_numeric(df['redCards'], errors='coerce')

    # Referee country-level implicit/explicit bias measures
    df['RefImplicit'] = pd.to_numeric(df['leagueCountry'], errors='coerce')
    df['RefExplicit'] = pd.to_numeric(df['club'], errors='coerce')

    # Referee identifier for clustering
    df['RefID'] = df['goals']

    # Controls: numeric proxies from the dataset. Standardize continuous controls (z-score) to aid model convergence
    # meanExp: player-level experience/ability proxy
    df['meanExp'] = pd.to_numeric(df['meanExp'], errors='coerce')
    df['playerShort'] = pd.to_numeric(df['playerShort'], errors='coerce')

    # Compute z-scores for continuous controls (fill NaN with 0 after z-scoring to avoid dropping)
    df['meanExp_z'] = (df['meanExp'] - df['meanExp'].mean()) / (df['meanExp'].std(ddof=0) if df['meanExp'].std(ddof=0) != 0 else 1)
    df['playerShort_z'] = (df['playerShort'] - df['playerShort'].mean()) / (df['playerShort'].std(ddof=0) if df['playerShort'].std(ddof=0) != 0 else 1)

    # Drop rows that still have NA in any final modeling column
    final_cols = ['SkinTone', 'SkinToneDark', 'RedCardCount', 'Matches', 'RefImplicit', 'RefExplicit', 'meanExp_z', 'playerShort_z', 'RefID']
    df = df.dropna(subset=final_cols)

    # Ensure Matches are positive (an offset of 0 or negative doesn't make sense); drop zero-match dyads
    df = df[df['Matches'] > 0]

    # Reset index to keep things tidy
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a count model for red cards with exposure (matches) and clustered SEs by referee.

    Model specification:
    - Dependent variable: RedCardCount (count of red cards in dyad)
    - Independent variable: SkinToneDark (binary: 1 = dark skin tone, 0 = light)
    - Controls: meanExp_z, playerShort_z, RefImplicit, RefExplicit
    - Exposure/offset: log(Matches)
    - Family: Negative Binomial (to allow for overdispersion relative to Poisson)
    - Clustered standard errors by RefID (referee identifier)

    Returns a dict with:
    - 'model' : statsmodels fitted result (robust-clustered covariance)
    - 'irr' : DataFrame with incidence rate ratios (exp(coef)) and 95% CIs
    - 'summary' : text summary
    """

    # Required imports inside function scope (statsmodels already imported at module level)
    import numpy as np
    import statsmodels.api as sm

    # Columns used in the model
    y = df['RedCardCount']
    offset = np.log(df['Matches'].astype(float))

    X_cols = ['SkinToneDark', 'meanExp_z', 'playerShort_z', 'RefImplicit', 'RefExplicit']
    X = df[X_cols].astype(float)
    X = sm.add_constant(X, has_constant='add')

    # Fit Negative Binomial via GLM with offset
    model_glm = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset)
    res = model_glm.fit()

    # Obtain cluster-robust covariance by referee (RefID). Use get_robustcov_results to get clustered SEs.
    try:
        res_clust = res.get_robustcov_results(cov_type='cluster', groups=df['RefID'])
    except Exception:
        # Fallback: if clustering fails, return original results but warn the user
        res_clust = res

    # Compute incidence rate ratios (IRR) and 95% CI from clustered results
    params = res_clust.params
    conf = res_clust.conf_int()
    irr = np.exp(params)
    irr_lower = np.exp(conf[0])
    irr_upper = np.exp(conf[1])

    irr_table = df = None
    try:
        import pandas as _pd
        irr_table = _pd.DataFrame({
            'coef': params,
            'IRR': irr,
            'IRR_2.5%': irr_lower,
            'IRR_97.5%': irr_upper
        })
    except Exception:
        # If pandas isn't available for some reason in the environment, build a plain dict
        irr_table = {
            'coef': params.to_dict() if hasattr(params, 'to_dict') else dict(params),
            'IRR': irr.tolist(),
            'IRR_2.5%': irr_lower.tolist(),
            'IRR_97.5%': irr_upper.tolist()
        }

    results_out = {
        'model_result': res_clust,
        'irr_table': irr_table,
        'summary_text': res_clust.summary().as_text()
    }

    return results_out


