import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('soccer.csv')

# Create skin tone average

df['skin_tone'] = df[['rater1', 'rater2']].mean(axis=1)

# Keep rows with skin tone and games > 0

d = df.loc[df['skin_tone'].notna() & (df['games'] > 0)].copy()

# Basic rate per game

d['red_rate'] = d['redCards'] / d['games']

# Define light vs dark using midpoint (<=0.5 light, >0.5 dark)

d['dark_skin'] = (d['skin_tone'] > 0.5).astype(int)

# Group comparison

group_stats = d.groupby('dark_skin').agg(
    n=('redCards', 'size'),
    total_red=('redCards', 'sum'),
    total_games=('games', 'sum'),
    mean_rate=('red_rate', 'mean')
)
group_stats['rate_per_game'] = group_stats['total_red'] / group_stats['total_games']

# Poisson regression with offset for games

d['log_games'] = np.log(d['games'])

# Model 1: skin tone only

model1 = smf.glm(
    'redCards ~ skin_tone',
    data=d,
    family=sm.families.Poisson(),
    offset=d['log_games']
).fit(cov_type='HC0')

# Model 2: with controls (leagueCountry, position, height, weight)

model2 = smf.glm(
    'redCards ~ skin_tone + C(leagueCountry) + C(position) + height + weight',
    data=d,
    family=sm.families.Poisson(),
    offset=d['log_games']
).fit(cov_type='HC0')

# Extract effect sizes

def poisson_rr(model):
    coef = model.params['skin_tone']
    se = model.bse['skin_tone']
    rr = np.exp(coef)
    ci_low = np.exp(coef - 1.96 * se)
    ci_high = np.exp(coef + 1.96 * se)
    pval = model.pvalues['skin_tone']
    return rr, ci_low, ci_high, pval

rr1, ci1_low, ci1_high, p1 = poisson_rr(model1)
rr2, ci2_low, ci2_high, p2 = poisson_rr(model2)

# Simple two-sample rate comparison using Poisson rate ratio (approx)

if 0 in group_stats.index and 1 in group_stats.index:
    light_red = group_stats.loc[0, 'total_red']
    dark_red = group_stats.loc[1, 'total_red']
    light_games = group_stats.loc[0, 'total_games']
    dark_games = group_stats.loc[1, 'total_games']
    rate_ratio = (dark_red / dark_games) / (light_red / light_games) if light_red > 0 else np.nan
    se_log = np.sqrt(1 / dark_red + 1 / light_red) if (dark_red > 0 and light_red > 0) else np.nan
    if np.isfinite(se_log):
        ci_low = np.exp(np.log(rate_ratio) - 1.96 * se_log)
        ci_high = np.exp(np.log(rate_ratio) + 1.96 * se_log)
    else:
        ci_low = np.nan
        ci_high = np.nan
else:
    rate_ratio = np.nan
    ci_low = np.nan
    ci_high = np.nan

# Save results to JSON for later use

results = {
    'group_stats': group_stats.reset_index().to_dict(orient='records'),
    'model1': {
        'rr': rr1,
        'ci_low': ci1_low,
        'ci_high': ci1_high,
        'pval': p1,
        'coef': model1.params['skin_tone']
    },
    'model2': {
        'rr': rr2,
        'ci_low': ci2_low,
        'ci_high': ci2_high,
        'pval': p2,
        'coef': model2.params['skin_tone']
    },
    'rate_ratio': rate_ratio,
    'rate_ratio_ci': [ci_low, ci_high]
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('done')
