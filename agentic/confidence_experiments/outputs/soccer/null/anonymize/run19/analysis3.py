import pandas as pd
import numpy as np
import statsmodels.api as sm


df = pd.read_csv('soccer.csv')
skin = df[['feature18', 'feature19']].mean(axis=1, skipna=True)
df = df.assign(skin_tone=skin)

dyad = df.dropna(subset=['skin_tone', 'feature16', 'feature9']).copy()

# define light/dark categories at dyad level
light = dyad[dyad['skin_tone'] <= 0.25]
dark = dyad[dyad['skin_tone'] >= 0.75]

print('dyad light rows', len(light), 'dark rows', len(dark))

# rates per game
for name, grp in [('light', light), ('dark', dark), ('mid', dyad[(dyad['skin_tone']>0.25)&(dyad['skin_tone']<0.75)])]:
    rate = (grp['feature16'].sum() / grp['feature9'].sum()) if grp['feature9'].sum()>0 else np.nan
    print(name, 'red cards per game', rate, 'total red cards', grp['feature16'].sum(), 'total games', grp['feature9'].sum())

# Poisson regression with dark indicator
ld = dyad[(dyad['skin_tone'] <= 0.25) | (dyad['skin_tone'] >= 0.75)].copy()
ld['dark_indicator'] = (ld['skin_tone'] >= 0.75).astype(int)
X = sm.add_constant(ld['dark_indicator'])
y = ld['feature16']
offset = np.log(ld['feature9'])
model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
res = model.fit(cov_type='HC0')
print('\nPoisson GLM dyad light vs dark')
print(res.summary())
coef = res.params['dark_indicator']
se = res.bse['dark_indicator']
irr = np.exp(coef)
ci_low = np.exp(coef - 1.96*se)
ci_high = np.exp(coef + 1.96*se)
print('IRR dark vs light:', irr, '95% CI', ci_low, ci_high)

# Mann-Whitney on per-dyad red_per_game (sensitive to game count)
ld['red_per_game'] = ld['feature16'] / ld['feature9']
try:
    from scipy.stats import mannwhitneyu
    stat, p = mannwhitneyu(ld[ld['dark_indicator']==1]['red_per_game'], ld[ld['dark_indicator']==0]['red_per_game'], alternative='two-sided')
    print('Mann-Whitney p-value:', p)
except Exception as e:
    print('Mann-Whitney failed', e)
