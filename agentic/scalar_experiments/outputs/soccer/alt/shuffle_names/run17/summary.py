import os, sys
cwd=os.getcwd(); sys.path=[p for p in sys.path if p not in ("", cwd)]
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load
_df=pd.read_csv('soccer.csv')

skin_mean = pd.concat([_df['rater1'], _df['nExp']], axis=1).mean(axis=1, skipna=True)
total_red = _df['meanExp'] + _df['yellowCards']
games = _df['redCards']

analysis = pd.DataFrame({
    'skin_mean': skin_mean,
    'total_red': total_red,
    'games': games,
})
analysis = analysis.dropna(subset=['skin_mean'])
analysis = analysis[analysis['games'] > 0]
analysis['skin_cat'] = pd.cut(
    analysis['skin_mean'],
    bins=[-np.inf, 0.25, 0.5, 0.75, np.inf],
    labels=['light', 'medium_light', 'medium_dark', 'dark']
)
ld = analysis[analysis['skin_cat'].isin(['light','dark'])].copy()
ld['dark'] = (ld['skin_cat'] == 'dark').astype(int)

# rates
rates = ld.groupby('skin_cat').apply(lambda g: pd.Series({
    'n': len(g),
    'total_reds': g['total_red'].sum(),
    'games': g['games'].sum(),
    'rate_per_game': g['total_red'].sum() / g['games'].sum()
}))

# model
X = sm.add_constant(ld['dark'])
model = sm.GLM(ld['total_red'], X, family=sm.families.Poisson(), offset=np.log(ld['games']))
res = model.fit(cov_type='HC0')

beta = res.params['dark']
se = res.bse['dark']
irr = np.exp(beta)
ci_low = np.exp(beta - 1.96*se)
ci_high = np.exp(beta + 1.96*se)

print('rates')
print(rates)
print('beta', beta, 'se', se, 'p', res.pvalues['dark'])
print('irr', irr, 'ci', ci_low, ci_high)

# continuous
analysis['skin_mean_centered'] = analysis['skin_mean'] - analysis['skin_mean'].mean()
Xc = sm.add_constant(analysis['skin_mean_centered'])
res_c = sm.GLM(analysis['total_red'], Xc, family=sm.families.Poisson(), offset=np.log(analysis['games'])).fit(cov_type='HC0')

beta_c = res_c.params['skin_mean_centered']
se_c = res_c.bse['skin_mean_centered']
irr_c = np.exp(beta_c)
ci_low_c = np.exp(beta_c - 1.96*se_c)
ci_high_c = np.exp(beta_c + 1.96*se_c)
print('cont beta', beta_c, 'se', se_c, 'p', res_c.pvalues['skin_mean_centered'])
print('cont irr', irr_c, 'ci', ci_low_c, ci_high_c)

