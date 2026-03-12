import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
_df = pd.read_csv('soccer.csv')

# Compute average skin tone (0=very light, 1=very dark)
_df['skin'] = _df[['rater1', 'rater2']].mean(axis=1)

# Create groups: light (<=0.25), dark (>=0.75), mid (otherwise)
_df['skin_group'] = np.where(_df['skin'] <= 0.25, 'light',
                     np.where(_df['skin'] >= 0.75, 'dark', 'mid'))

# Keep only light and dark for primary comparison
sub = _df[_df['skin_group'].isin(['light', 'dark'])].copy()

# Basic aggregates
agg = sub.groupby('skin_group').agg(
    players=('playerShort', 'nunique'),
    dyads=('playerShort', 'size'),
    red_cards=('redCards', 'sum'),
    games=('games', 'sum')
)
agg['red_per_100_games'] = 100 * agg['red_cards'] / agg['games']

# Poisson regression with offset(log games)
sub['dark'] = (sub['skin_group'] == 'dark').astype(int)
sub['log_games'] = np.log(sub['games'])

# GLM Poisson
model = sm.GLM(sub['redCards'], sm.add_constant(sub['dark']),
               family=sm.families.Poisson(), offset=sub['log_games'])
res = model.fit(cov_type='cluster', cov_kwds={'groups': sub['playerShort']})

# Extract effect for dark
coef = res.params['dark']
se = res.bse['dark']
irr = float(np.exp(coef))

# 95% CI for IRR
ci_low = float(np.exp(coef - 1.96 * se))
ci_high = float(np.exp(coef + 1.96 * se))

p_value = float(res.pvalues['dark'])

# Also compute rate ratio from aggregates
rate_light = agg.loc['light', 'red_per_100_games']
rate_dark = agg.loc['dark', 'red_per_100_games']
rate_ratio = float(rate_dark / rate_light) if rate_light > 0 else np.nan

# Prepare results for reporting
results = {
    'n_total': int(len(_df)),
    'n_light_dark': int(len(sub)),
    'agg': agg.reset_index().to_dict(orient='records'),
    'rate_light_per_100_games': float(rate_light),
    'rate_dark_per_100_games': float(rate_dark),
    'rate_ratio_dark_vs_light': rate_ratio,
    'poisson_irr_dark_vs_light': irr,
    'poisson_ci95_low': ci_low,
    'poisson_ci95_high': ci_high,
    'poisson_p_value': p_value,
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)
