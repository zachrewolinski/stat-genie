import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('soccer.csv')

# Prepare skin tone measure
for col in ['rater1', 'rater2']:
    if col not in df.columns:
        raise ValueError(f"Missing {col}")

skin_mean = df[['rater1', 'rater2']].mean(axis=1, skipna=True)

# Define dark/light groups (exclude middle/neutral 0.5)

def skin_group(x):
    if pd.isna(x):
        return np.nan
    if x > 0.5:
        return 'dark'
    if x < 0.5:
        return 'light'
    return 'neutral'


df = df.copy()

df['skin_mean'] = skin_mean

df['skin_group'] = df['skin_mean'].apply(skin_group)

# Filter to dark/light with valid exposure
analysis_df = df[(df['skin_group'].isin(['dark', 'light'])) & (df['games'] > 0)]

# Summary rates
summary = analysis_df.groupby('skin_group').agg(
    red_cards=('redCards', 'sum'),
    games=('games', 'sum'),
    dyads=('redCards', 'size'),
    players=('playerShort', pd.Series.nunique)
).reset_index()
summary['rate_per_game'] = summary['red_cards'] / summary['games']
summary['rate_per_100_games'] = summary['rate_per_game'] * 100

print('Summary by skin group (dyad-level):')
print(summary)

# Poisson regression with offset and cluster-robust SE by player
analysis_df = analysis_df.copy()
analysis_df['dark'] = (analysis_df['skin_group'] == 'dark').astype(int)
analysis_df['log_games'] = np.log(analysis_df['games'])

model = smf.glm(
    formula='redCards ~ dark',
    data=analysis_df,
    family=sm.families.Poisson(),
    offset=analysis_df['log_games']
)
res = model.fit(cov_type='cluster', cov_kwds={'groups': analysis_df['playerShort']})

coef = res.params['dark']
se = res.bse['dark']
pval = res.pvalues['dark']

irr = np.exp(coef)
ci_low, ci_high = np.exp(res.conf_int().loc['dark'])

print('\nDyad-level Poisson (cluster-robust by player):')
print(res.summary().tables[1])
print(f"IRR (dark vs light): {irr:.3f} (95% CI {ci_low:.3f}, {ci_high:.3f}), p={pval:.4g}")

# Player-level aggregation
player_df = df[df['skin_mean'].notna()].copy()
player_df = player_df.groupby('playerShort').agg(
    skin_mean=('skin_mean', 'mean'),
    games=('games', 'sum'),
    redCards=('redCards', 'sum')
).reset_index()

player_df['skin_group'] = player_df['skin_mean'].apply(skin_group)
player_df = player_df[(player_df['skin_group'].isin(['dark', 'light'])) & (player_df['games'] > 0)]
player_df['dark'] = (player_df['skin_group'] == 'dark').astype(int)
player_df['log_games'] = np.log(player_df['games'])

player_model = smf.glm(
    formula='redCards ~ dark',
    data=player_df,
    family=sm.families.Poisson(),
    offset=player_df['log_games']
)
player_res = player_model.fit()

p_coef = player_res.params['dark']
p_pval = player_res.pvalues['dark']
p_irr = np.exp(p_coef)
p_ci_low, p_ci_high = np.exp(player_res.conf_int().loc['dark'])

print('\nPlayer-level Poisson:')
print(player_res.summary().tables[1])
print(f"IRR (dark vs light): {p_irr:.3f} (95% CI {p_ci_low:.3f}, {p_ci_high:.3f}), p={p_pval:.4g}")

# Rate comparison with approximate Poisson rate ratio test (dyad-level aggregated)
from scipy.stats import norm

# Get counts and exposure
light = summary[summary['skin_group'] == 'light'].iloc[0]
dark = summary[summary['skin_group'] == 'dark'].iloc[0]

rate1 = dark['red_cards'] / dark['games']
rate2 = light['red_cards'] / light['games']
rate_ratio = rate1 / rate2

se_log = np.sqrt(1 / dark['red_cards'] + 1 / light['red_cards'])
z = np.log(rate_ratio) / se_log
p_value = 2 * (1 - norm.cdf(abs(z)))
rate_ci_low = np.exp(np.log(rate_ratio) - 1.96 * se_log)
rate_ci_high = np.exp(np.log(rate_ratio) + 1.96 * se_log)

print('\nPoisson rate ratio test (dark vs light):')
print(f"Rate ratio: {rate_ratio:.3f}, 95% CI ({rate_ci_low:.3f}, {rate_ci_high:.3f}), p={p_value:.4g}")

# Save key results to a small json for later reference
results = {
    'summary': summary.to_dict(orient='records'),
    'dyad_irr': float(irr),
    'dyad_ci_low': float(ci_low),
    'dyad_ci_high': float(ci_high),
    'dyad_p': float(pval),
    'player_irr': float(p_irr),
    'player_ci_low': float(p_ci_low),
    'player_ci_high': float(p_ci_high),
    'player_p': float(p_pval),
    'rate_ratio': float(rate_ratio),
    'rate_ratio_ci_low': float(rate_ci_low),
    'rate_ratio_ci_high': float(rate_ci_high),
    'rate_ratio_p': float(p_value)
}

import json
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\nWrote analysis_results.json')
