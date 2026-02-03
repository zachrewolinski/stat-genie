import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
# Note: column names appear shuffled relative to the original dataset description,
# but distributions indicate rater1/nExp are skin tone ratings (0-1), and redCards
# is the number of games (min 1, max ~47). Low-count columns are used as red cards.

df = pd.read_csv('soccer.csv')

# Skin tone: average of two raters (0=very light, 1=very dark)
df['skin_mean'] = df[['rater1', 'nExp']].mean(axis=1)

# Define light vs dark using clear extremes
# Light: <= 0.25 (very light/light), Dark: >= 0.75 (dark/very dark)
df = df[df['skin_mean'].notna()]
light_mask = df['skin_mean'] <= 0.25
dark_mask = df['skin_mean'] >= 0.75

# Exposure: number of games in the player-referee dyad
# The column named redCards is the only one with a 1-47 range, consistent with games.
df = df[df['redCards'] > 0]

# Red cards: sum of two rare red-card-like columns
# (yellowCards and meanExp are both extremely low-count; together they represent red cards)
df['red_total'] = df['yellowCards'] + df['meanExp']

# Restrict to light/dark groups only
sub = df[light_mask | dark_mask].copy()
sub['dark'] = (sub['skin_mean'] >= 0.75).astype(int)

# Aggregate rates
agg = sub.groupby('dark')[['red_total', 'redCards']].sum()
rate_light = agg.loc[0, 'red_total'] / agg.loc[0, 'redCards']
rate_dark = agg.loc[1, 'red_total'] / agg.loc[1, 'redCards']

# Rate ratio and 95% CI (Poisson counts)
import math
k_dark = agg.loc[1, 'red_total']
k_light = agg.loc[0, 'red_total']
E_dark = agg.loc[1, 'redCards']
E_light = agg.loc[0, 'redCards']
rr = (k_dark / E_dark) / (k_light / E_light)
se_log_rr = math.sqrt(1 / k_dark + 1 / k_light)
ci_low = math.exp(math.log(rr) - 1.96 * se_log_rr)
ci_high = math.exp(math.log(rr) + 1.96 * se_log_rr)

# Poisson regression with offset for games
X = sm.add_constant(sub['dark'])
model = sm.GLM(sub['red_total'], X, family=sm.families.Poisson(), offset=np.log(sub['redCards']))
res = model.fit(cov_type='HC0')

print('Light rate per game:', rate_light)
print('Dark rate per game:', rate_dark)
print('Rate ratio (dark/light):', rr)
print('95% CI:', (ci_low, ci_high))
print('Poisson regression (dark indicator):')
print(res.summary().tables[1])

# Save key results for conclusion
results = {
    'rate_light': rate_light,
    'rate_dark': rate_dark,
    'rr': rr,
    'ci_low': ci_low,
    'ci_high': ci_high,
    'p_value': float(res.pvalues['dark']) if 'dark' in res.pvalues else float(res.pvalues.iloc[1])
}

pd.DataFrame([results]).to_csv('analysis_results.csv', index=False)
