import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm

# Load data
path = 'soccer.csv'
df = pd.read_csv(path)

# Compute average skin tone (0-1) across two raters
skin = df[['feature18', 'feature19']].mean(axis=1)

df = df.copy()
df['skin_tone'] = skin

# Keep rows with necessary data
# feature9: games in dyad, feature16: red cards

df = df.dropna(subset=['skin_tone', 'feature9', 'feature16'])

df = df[df['feature9'] > 0]

# Poisson regression: red_cards ~ skin_tone + offset(log(games))
X = sm.add_constant(df['skin_tone'])
offset = np.log(df['feature9'].astype(float))

y = df['feature16']

poisson_model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
poisson_res = poisson_model.fit()

# Define light vs dark groups from skin tone (average of 2 raters on 5-point scale normalized to 0-1)
# light: <=0.25, dark: >=0.75 (approx 1 and 5 on original scale)

df['skin_group'] = pd.cut(
    df['skin_tone'],
    bins=[-0.01, 0.25, 0.75, 1.01],
    labels=['light', 'medium', 'dark']
)

# Group stats: total red cards, total games, dyad count

group_stats = df.groupby('skin_group').agg(
    red_cards=('feature16', 'sum'),
    games=('feature9', 'sum'),
    dyads=('skin_group', 'size')
)

group_stats['red_cards_per_game'] = group_stats['red_cards'] / group_stats['games']

# Rate ratio for dark vs light (pooled across dyads)
light = group_stats.loc['light']
dark = group_stats.loc['dark']

if light['red_cards'] > 0 and dark['red_cards'] > 0:
    rr = (dark['red_cards'] / dark['games']) / (light['red_cards'] / light['games'])
    var_log_rr = 1 / dark['red_cards'] + 1 / light['red_cards']
    se_log_rr = np.sqrt(var_log_rr)
    z = np.log(rr) / se_log_rr
    p_rr = 2 * (1 - norm.cdf(abs(z)))
    ci_low = np.exp(np.log(rr) - 1.96 * se_log_rr)
    ci_high = np.exp(np.log(rr) + 1.96 * se_log_rr)
else:
    rr = np.nan
    p_rr = np.nan
    ci_low = np.nan
    ci_high = np.nan

# Model-based rate ratio for skin tone 1.0 vs 0.0
beta = poisson_res.params
rate_light = np.exp(beta['const'] + beta['skin_tone'] * 0)
rate_dark = np.exp(beta['const'] + beta['skin_tone'] * 1)
rate_ratio_model = rate_dark / rate_light
p_skin = poisson_res.pvalues['skin_tone']

# Save key results to a small JSON for reuse
results = {
    'n_rows_used': int(len(df)),
    'coef_skin_tone': float(poisson_res.params['skin_tone']),
    'p_skin_tone': float(p_skin),
    'rate_ratio_model_1_vs_0': float(rate_ratio_model),
    'group_stats': {
        'light': {
            'red_cards': float(light['red_cards']),
            'games': float(light['games']),
            'dyads': int(light['dyads']),
            'red_cards_per_game': float(light['red_cards_per_game']),
        },
        'dark': {
            'red_cards': float(dark['red_cards']),
            'games': float(dark['games']),
            'dyads': int(dark['dyads']),
            'red_cards_per_game': float(dark['red_cards_per_game']),
        },
        'medium': {
            'red_cards': float(group_stats.loc['medium', 'red_cards']),
            'games': float(group_stats.loc['medium', 'games']),
            'dyads': int(group_stats.loc['medium', 'dyads']),
            'red_cards_per_game': float(group_stats.loc['medium', 'red_cards_per_game']),
        },
    },
    'rr_dark_vs_light': float(rr),
    'rr_p_value': float(p_rr),
    'rr_ci_low': float(ci_low),
    'rr_ci_high': float(ci_high),
}

import json
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
