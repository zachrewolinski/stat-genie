import json
import numpy as np
import pandas as pd
import statsmodels.api as sm


df = pd.read_csv('soccer.csv')

# Row-level skin tone (mean of available rater scores)
raters = df[['rater1', 'rater2']]
skin_tone = raters.mean(axis=1)

df = df.assign(skin_tone=skin_tone)

df = df[df['skin_tone'].notna()].copy()
if 'games' in df.columns:
    df = df[df['games'] > 0].copy()


def fit_poisson(y, X, offset):
    model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
    return model.fit(cov_type='HC3')


def summarize_continuous(res, var_name):
    coef = res.params[var_name]
    se = res.bse[var_name]
    rr = float(np.exp(coef))
    ci = np.exp(coef + np.array([-1, 1]) * 1.96 * se)
    p = float(res.pvalues[var_name])
    return {
        'coef': float(coef),
        'se': float(se),
        'rate_ratio': rr,
        'rr_ci_low': float(ci[0]),
        'rr_ci_high': float(ci[1]),
        'p_value': p,
        'n_obs': int(res.nobs)
    }


def summarize_binary(data, dark_mask, light_mask):
    subset = data[dark_mask | light_mask].copy()
    subset['dark'] = np.where(subset.index.isin(data[dark_mask].index), 1, 0)
    # Poisson rate model with offset
    X = sm.add_constant(subset['dark'])
    res = fit_poisson(subset['redCards'], X, np.log(subset['games']))
    coef = res.params['dark']
    se = res.bse['dark']
    rr = float(np.exp(coef))
    ci = np.exp(coef + np.array([-1, 1]) * 1.96 * se)
    p = float(res.pvalues['dark'])

    def rate_stats(sub):
        total_red = sub['redCards'].sum()
        total_games = sub['games'].sum()
        rate = total_red / total_games if total_games > 0 else np.nan
        return int(total_red), int(total_games), float(rate)

    dark_red, dark_games, dark_rate = rate_stats(data[dark_mask])
    light_red, light_games, light_rate = rate_stats(data[light_mask])

    return {
        'n_dark': int(dark_mask.sum()),
        'n_light': int(light_mask.sum()),
        'dark_red': dark_red,
        'dark_games': dark_games,
        'dark_rate': dark_rate,
        'light_red': light_red,
        'light_games': light_games,
        'light_rate': light_rate,
        'rate_ratio': rr,
        'rr_ci_low': float(ci[0]),
        'rr_ci_high': float(ci[1]),
        'p_value': p
    }

# Dyad-level continuous
X_dyad = sm.add_constant(df['skin_tone'])
res_dyad = fit_poisson(df['redCards'], X_dyad, np.log(df['games']))
summary_dyad = summarize_continuous(res_dyad, 'skin_tone')

# Dyad-level dark vs light using extreme cutoffs (<=0.25 light, >=0.75 dark)
dyad_binary = summarize_binary(
    df,
    dark_mask=df['skin_tone'] >= 0.75,
    light_mask=df['skin_tone'] <= 0.25
)

# Player-level aggregation using median skin tone per player
player = df.groupby('playerShort', as_index=False).agg(
    skin_tone_median=('skin_tone', 'median'),
    redCards=('redCards', 'sum'),
    games=('games', 'sum')
)
player = player[(player['games'] > 0) & (player['skin_tone_median'].notna())].copy()

X_player = sm.add_constant(player['skin_tone_median'])
res_player = fit_poisson(player['redCards'], X_player, np.log(player['games']))
summary_player = summarize_continuous(res_player, 'skin_tone_median')

player_binary = summarize_binary(
    player,
    dark_mask=player['skin_tone_median'] >= 0.75,
    light_mask=player['skin_tone_median'] <= 0.25
)

summary = {
    'dyad_continuous': summary_dyad,
    'dyad_binary_extremes': dyad_binary,
    'player_continuous_median': summary_player,
    'player_binary_extremes': player_binary,
    'dyad_count': int(len(df)),
    'player_count': int(len(player))
}

with open('analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
