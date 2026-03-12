import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest

# Load data
csv_path = 'soccer.csv'

df = pd.read_csv(csv_path)

# Skin tone average
# rater1 and rater2 are normalized 0-1; average where available
skin = df[['rater1','rater2']].mean(axis=1)

df = df.copy()
df['skin_avg'] = skin

# Ensure numeric
for col in ['redCards','games']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Filter valid rows with skin and games>0
analysis_df = df[(df['skin_avg'].notna()) & (df['games'] > 0)].copy()

# Define light vs dark groups: light <=0.25, dark >=0.75 (extremes) and mid elsewhere
analysis_df['skin_group'] = pd.cut(
    analysis_df['skin_avg'],
    bins=[-np.inf, 0.25, 0.75, np.inf],
    labels=['light','mid','dark']
)

# Compute rates per game
analysis_df['red_per_game'] = analysis_df['redCards'] / analysis_df['games']

# Descriptive stats
rate_stats = analysis_df.groupby('skin_group')['red_per_game'].agg(['mean','count'])

# Also compute total red cards and games by group
agg = analysis_df.groupby('skin_group').agg(redCards=('redCards','sum'), games=('games','sum'))
agg['rate'] = agg['redCards'] / agg['games']

# Two-sample test for proportions (rate) using counts of redCards over games for light vs dark
# This treats each game as Bernoulli, which is approximate.
if 'light' in agg.index and 'dark' in agg.index:
    count = np.array([agg.loc['dark','redCards'], agg.loc['light','redCards']])
    nobs = np.array([agg.loc['dark','games'], agg.loc['light','games']])
    zstat, pval = proportions_ztest(count, nobs)
else:
    zstat, pval = np.nan, np.nan

# Poisson regression with offset log(games)
# Use continuous skin_avg to capture monotonic relationship
# Clustered SE by playerShort
analysis_df['log_games'] = np.log(analysis_df['games'])

X = sm.add_constant(analysis_df['skin_avg'])
model = sm.GLM(analysis_df['redCards'], X, family=sm.families.Poisson(), offset=analysis_df['log_games'])

res = model.fit()

# Clustered SE by playerShort
# If playerShort missing, fallback to default
try:
    res_cluster = model.fit(cov_type='cluster', cov_kwds={'groups': analysis_df['playerShort']})
    res_use = res_cluster
except Exception:
    res_use = res

coef = res_use.params['skin_avg']
se = res_use.bse['skin_avg']
p_value = res_use.pvalues['skin_avg']

# Effect: rate ratio for dark (0.75) vs light (0.25) and for 1.0 vs 0.0
rate_ratio_075_025 = np.exp(coef * (0.75-0.25))
rate_ratio_1_0 = np.exp(coef * (1.0-0.0))

# Output summary
print('N rows total:', len(df))
print('N used:', len(analysis_df))
print('\nRate stats (mean red_per_game):\n', rate_stats)
print('\nAggregate counts:\n', agg)
print('\nZ test dark vs light: z=%.3f p=%.4g' % (zstat, pval))
print('\nPoisson regression (offset log games) with skin_avg:')
print(res_use.summary())
print('\nCoef skin_avg:', coef, 'SE:', se, 'p:', p_value)
print('Rate ratio dark(0.75) vs light(0.25):', rate_ratio_075_025)
print('Rate ratio 1.0 vs 0.0:', rate_ratio_1_0)
