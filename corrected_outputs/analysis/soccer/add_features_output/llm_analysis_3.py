from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/soccer/add_features_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Keep only rows with the essential variables present
    df = df.dropna(subset=['rater1', 'rater2', 'games', 'redCards'])

    # Compute average rater score (rater variables are normalized to 0-1)
    df['avg_rater'] = (df['rater1'] + df['rater2']) / 2.0

    # Categorize into Dark vs Light. Exclude intermediates to create a clear comparison.
    def _skin_category(x):
        if pd.isna(x):
            return np.nan
        if x >= 0.75:
            return 'Dark'
        if x <= 0.25:
            return 'Light'
        return np.nan

    df['SkinBin'] = df['avg_rater'].apply(_skin_category)

    # Keep only clear Dark or Light cases
    df = df.dropna(subset=['SkinBin'])

    # Binary numeric coding used in the model: 1 = Dark, 0 = Light
    df['SkinDark'] = (df['SkinBin'] == 'Dark').astype(int)

    # Remove dyads with zero games (no exposure) because they provide no exposure for rates
    df = df[df['games'] > 0]

    # Offset for rate models: log of number of games in the dyad
    df['logGames'] = np.log(df['games'].astype(float))

    # Ensure numeric controls are numeric (coerce non-numeric to NaN)
    numeric_controls = ['age', 'height', 'weight', 'yellowCards', 'meanIAT', 'meanExp']
    for col in numeric_controls:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing key numeric controls (so model will have a complete set of controls).
    # Depending on analysis goals, you could instead impute -- here we drop to keep the model straightforward.
    required_controls = ['age', 'height', 'yellowCards', 'meanIAT', 'meanExp']
    present_required = [c for c in required_controls if c in df.columns]
    if present_required:
        df = df.dropna(subset=present_required)

    # Keep columns that will be used in modeling (this is the final dataframe returned)
    keep_cols = [
        'playerShort', 'player', 'club', 'leagueCountry', 'birthday', 'position',
        'games', 'redCards', 'yellowCards', 'avg_rater', 'SkinBin', 'SkinDark', 'logGames',
        'age', 'height', 'weight', 'meanIAT', 'meanExp', 'refNum'
    ]
    # Remove columns that are not present in the input but were listed above
    keep_cols = [c for c in keep_cols if c in df.columns]

    df = df[keep_cols]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Work on a copy
    data = df.copy()

    # Formula: model redCards count with exposure (games). Use SkinDark (1=dark,0=light) as primary IV.
    # Controls: numeric player/referee and categorical position and leagueCountry.
    formula = (
        'redCards ~ SkinDark + age + height + weight + yellowCards + meanIAT + meanExp '
        '+ C(position) + C(leagueCountry)'
    )

    # Fit a Negative Binomial GLM with log(games) as an offset to model red-card rate per game.
    # Fall back to Poisson if NegativeBinomial fails.
    try:
        model_nb = smf.glm(formula=formula, data=data,
                           family=sm.families.NegativeBinomial(),
                           offset=data['logGames']).fit()
        base_model = model_nb
    except Exception:
        model_p = smf.glm(formula=formula, data=data,
                          family=sm.families.Poisson(),
                          offset=data['logGames']).fit()
        base_model = model_p

    # Compute cluster-robust standard errors clustered by referee (refNum). If refNum missing, return base_model.
    results = base_model
    if 'refNum' in data.columns:
        try:
            clustered = results.get_robustcov_results(cov_type='cluster', groups=data['refNum'])
            results = clustered
        except Exception:
            # If robustcov fails, keep original
            results = base_model

    # Print summary for quick inspection (the function returns the results object)
    try:
        print(results.summary())
    except Exception:
        pass

    return results


