import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
csv_path = 'soccer.csv'
df = pd.read_csv(csv_path)

# Compute skin tone mean
skin = df[['rater1','rater2']].mean(axis=1)
df = df.assign(skin_mean=skin)

# Filter rows with skin ratings and games >0
sub = df[df['skin_mean'].notna() & df['games'].notna() & (df['games']>0) & df['redCards'].notna()].copy()

# Categorize skin tone
sub['skin_group'] = pd.cut(
    sub['skin_mean'],
    bins=[-np.inf, 0.25, 0.5, 0.75, np.inf],
    labels=['light','medium','dark','very_dark'],
    right=True
)
# Merge dark categories (>=0.75)
sub['dark'] = (sub['skin_mean'] >= 0.75).astype(int)
sub['light'] = (sub['skin_mean'] <= 0.25).astype(int)

# Use only light and dark groups for primary comparison
ld = sub[(sub['light']==1) | (sub['dark']==1)].copy()
ld['dark'] = (ld['skin_mean'] >= 0.75).astype(int)

# Summary rates
summary = ld.groupby('dark').agg(
    dyads=('redCards','size'),
    total_games=('games','sum'),
    total_red=('redCards','sum')
).reset_index()
summary['red_per_100_games'] = summary['total_red'] / summary['total_games'] * 100

print('Summary (light=0, dark=1):')
print(summary)

# Poisson regression with offset log(games)
# Add intercept
X = sm.add_constant(ld['dark'])
y = ld['redCards'].astype(float)
offset = np.log(ld['games'].astype(float))

model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
# Cluster by playerShort to account for repeated players
try:
    res = model.fit(cov_type='cluster', cov_kwds={'groups': ld['playerShort']})
except Exception:
    res = model.fit(cov_type='HC0')

print('\nPoisson regression results:')
print(res.summary())

# Extract IRR
coef = res.params['dark']
se = res.bse['dark']
irr = np.exp(coef)
ci_low = np.exp(coef - 1.96*se)
ci_high = np.exp(coef + 1.96*se)
print(f"IRR (dark vs light) = {irr:.3f} (95% CI {ci_low:.3f}, {ci_high:.3f}), p={res.pvalues['dark']:.4g}")

# Also check continuous skin_mean effect on rate
cont = sub.copy()
X2 = sm.add_constant(cont['skin_mean'])
model2 = sm.GLM(cont['redCards'].astype(float), X2, family=sm.families.Poisson(), offset=np.log(cont['games'].astype(float)))
try:
    res2 = model2.fit(cov_type='cluster', cov_kwds={'groups': cont['playerShort']})
except Exception:
    res2 = model2.fit(cov_type='HC0')

coef2 = res2.params['skin_mean']
se2 = res2.bse['skin_mean']
irr2 = np.exp(coef2)
ci2_low = np.exp(coef2 - 1.96*se2)
ci2_high = np.exp(coef2 + 1.96*se2)
print('\nContinuous skin_mean effect:')
print(res2.summary())
print(f"IRR per 1.0 increase in skin_mean = {irr2:.3f} (95% CI {ci2_low:.3f}, {ci2_high:.3f}), p={res2.pvalues['skin_mean']:.4g}")

# Simple rate difference using Poisson test (approx) and mean red cards per game at dyad level
ld['red_rate'] = ld['redCards'] / ld['games']
rate_means = ld.groupby('dark')['red_rate'].mean()
print('\nMean red_rate per dyad (not weighted):')
print(rate_means)

# Weighted rate per game (total red/total games already)

