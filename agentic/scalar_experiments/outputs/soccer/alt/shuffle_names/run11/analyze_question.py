import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

base = Path('.')
info = json.loads((base / 'info.json').read_text())
fields = info['data_desc']['fields']
col_desc = {f['column']: f['properties']['description'] for f in fields}

# Identify columns based on descriptions
rater1_col = None
rater2_col = None
red_cards_col = None
games_col = None

for col, desc in col_desc.items():
    d = desc.lower()
    if 'skin rating of photo by rater 1' in d:
        rater1_col = col
    if 'skin rating of photo by rater 2' in d:
        rater2_col = col
    if 'number of red cards player received from referee' in d:
        red_cards_col = col
    if 'number of games in the player-referee dyad' in d:
        games_col = col

if any(v is None for v in [rater1_col, rater2_col, red_cards_col, games_col]):
    raise RuntimeError('Failed to map required columns')

# Load data
_df = pd.read_csv(base / 'soccer.csv')

# Compute mean skin tone rating (0 to 1) using available rater scores
skin = _df[[rater1_col, rater2_col]].mean(axis=1, skipna=True)

# Add to dataframe
_df = _df.copy()
_df['skin_mean'] = skin

# Define light and dark groups
# Light: very light or light (<=0.25); Dark: dark or very dark (>=0.75)
_df['skin_group'] = np.where(_df['skin_mean'] <= 0.25, 'light', np.where(_df['skin_mean'] >= 0.75, 'dark', np.nan))

# Filter to light/dark with non-missing skin
sub = _df[_df['skin_group'].isin(['light', 'dark'])].copy()

# Outcome and exposure
sub['red_cards'] = sub[red_cards_col]
sub['games'] = sub[games_col]

# Remove any non-positive exposure
sub = sub[sub['games'] > 0].copy()

# Aggregate rates by group
agg = sub.groupby('skin_group').agg(
    red_cards=('red_cards', 'sum'),
    games=('games', 'sum'),
    dyads=('red_cards', 'size'),
)
agg['rate_per_game'] = agg['red_cards'] / agg['games']

print('Group aggregates:')
print(agg)

# Poisson regression with offset log(games)
# Predictor: dark vs light
sub['is_dark'] = (sub['skin_group'] == 'dark').astype(int)

X = sm.add_constant(sub['is_dark'])
# Use Poisson GLM
model = sm.GLM(sub['red_cards'], X, family=sm.families.Poisson(), offset=np.log(sub['games']))
res = model.fit(cov_type='HC0')  # robust SE

coef = res.params['is_dark']
se = res.bse['is_dark']
z = coef / se
p = 2 * (1 - stats.norm.cdf(abs(z)))

irr = math.exp(coef)
ci_low = math.exp(coef - 1.96 * se)
ci_high = math.exp(coef + 1.96 * se)

print('\nPoisson regression (robust SE):')
print(res.summary().tables[1])
print('IRR (dark vs light):', irr)
print('95% CI:', (ci_low, ci_high))
print('p-value:', p)

# Rate ratio using aggregated counts (Poisson rate ratio with normal approx)
# log rate ratio = log(r_dark / r_light)
rate_dark = agg.loc['dark', 'rate_per_game']
rate_light = agg.loc['light', 'rate_per_game']
rr = rate_dark / rate_light

# Approx SE for log rate ratio using Poisson counts
# var(log(rate)) ≈ 1/events; use red_cards counts
var_log_rr = 1/agg.loc['dark','red_cards'] + 1/agg.loc['light','red_cards']
se_log_rr = math.sqrt(var_log_rr)
ci_log_low = math.log(rr) - 1.96*se_log_rr
ci_log_high = math.log(rr) + 1.96*se_log_rr
ci_rr = (math.exp(ci_log_low), math.exp(ci_log_high))

print('\nAggregated rate ratio (dark/light):', rr)
print('95% CI:', ci_rr)

# Save a small json with key stats for later use
out = {
    'n_rows_total': int(_df.shape[0]),
    'n_light_dark': int(sub.shape[0]),
    'group_agg': agg.reset_index().to_dict(orient='records'),
    'poisson': {
        'coef': float(coef),
        'se': float(se),
        'z': float(z),
        'p': float(p),
        'irr': float(irr),
        'irr_ci_low': float(ci_low),
        'irr_ci_high': float(ci_high),
    },
    'rate_ratio': {
        'rr': float(rr),
        'rr_ci_low': float(ci_rr[0]),
        'rr_ci_high': float(ci_rr[1]),
    }
}

(base / 'analysis_results.json').write_text(json.dumps(out, indent=2))
print('\nSaved analysis_results.json')
