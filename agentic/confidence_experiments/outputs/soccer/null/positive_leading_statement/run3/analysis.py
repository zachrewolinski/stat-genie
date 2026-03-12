import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
csv_path = 'soccer.csv'

df = pd.read_csv(csv_path)

# Compute skin tone (mean of raters)
raters = df[['rater1', 'rater2']]
skin_tone = raters.mean(axis=1)

df = df.assign(skin_tone=skin_tone)

df = df[df['skin_tone'].notna()].copy()
# remove non-positive games
if 'games' in df.columns:
    df = df[df['games'] > 0].copy()

# Helper to fit Poisson with offset

def fit_poisson(y, X, offset):
    model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
    return model.fit(cov_type='HC3')

# Dyad-level Poisson: redCards ~ skin_tone + offset(log(games))
X_dyad = sm.add_constant(df['skin_tone'])
res_dyad = fit_poisson(df['redCards'], X_dyad, np.log(df['games']))

# Player-level aggregation
player = df.groupby('playerShort', as_index=False).agg(
    skin_tone=('skin_tone', 'mean'),
    redCards=('redCards', 'sum'),
    games=('games', 'sum')
)

player = player[(player['games'] > 0) & (player['skin_tone'].notna())].copy()
X_player = sm.add_constant(player['skin_tone'])
res_player = fit_poisson(player['redCards'], X_player, np.log(player['games']))

# Binary grouping thresholds

def group_rates(data, dark_mask, light_mask):
    dark = data[dark_mask]
    light = data[light_mask]
    def rate_stats(sub):
        total_red = sub['redCards'].sum()
        total_games = sub['games'].sum()
        rate = total_red / total_games if total_games > 0 else np.nan
        return total_red, total_games, rate
    dark_red, dark_games, dark_rate = rate_stats(dark)
    light_red, light_games, light_rate = rate_stats(light)
    # Poisson regression for rate ratio
    subset = data[dark_mask | light_mask].copy()
    subset['dark'] = np.where(dark_mask[dark_mask | light_mask], 1, 0)
    X = sm.add_constant(subset['dark'])
    res = fit_poisson(subset['redCards'], X, np.log(subset['games']))
    coef = res.params['dark']
    se = res.bse['dark']
    rr = np.exp(coef)
    ci = np.exp(coef + np.array([-1, 1]) * 1.96 * se)
    p = res.pvalues['dark']
    return {
        'dark_red': int(dark_red),
        'dark_games': int(dark_games),
        'dark_rate': dark_rate,
        'light_red': int(light_red),
        'light_games': int(light_games),
        'light_rate': light_rate,
        'rate_ratio': rr,
        'rr_ci_low': ci[0],
        'rr_ci_high': ci[1],
        'p_value': p,
        'n_dark': len(dark),
        'n_light': len(light)
    }

# Thresholds on player-level data
player_mid = group_rates(
    player,
    dark_mask=player['skin_tone'] > 0.5,
    light_mask=player['skin_tone'] < 0.5
)

player_strict = group_rates(
    player,
    dark_mask=player['skin_tone'] >= 0.75,
    light_mask=player['skin_tone'] <= 0.25
)

# Summaries

# For Poisson models, compute rate ratios per 1.0 increase in skin_tone

def summarize_poisson(res, label):
    coef = res.params['skin_tone']
    se = res.bse['skin_tone']
    rr = np.exp(coef)
    ci = np.exp(coef + np.array([-1, 1]) * 1.96 * se)
    p = res.pvalues['skin_tone']
    return {
        'label': label,
        'coef': coef,
        'se': se,
        'rate_ratio': rr,
        'rr_ci_low': ci[0],
        'rr_ci_high': ci[1],
        'p_value': p,
        'n_obs': int(res.nobs)
    }

summary = {
    'dyad_model': summarize_poisson(res_dyad, 'dyad_poisson'),
    'player_model': summarize_poisson(res_player, 'player_poisson'),
    'player_mid_threshold': player_mid,
    'player_strict_threshold': player_strict,
    'player_count': len(player),
    'dyad_count': len(df)
}

# Output to a temp json for inspection
import json
with open('analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
