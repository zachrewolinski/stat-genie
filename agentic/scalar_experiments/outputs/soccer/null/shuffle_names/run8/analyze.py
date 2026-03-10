import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('soccer.csv')

# Identify variables based on metadata/value ranges
# Skin tone ratings (normalized 0-1, in 0.25 increments)
skin1 = _df['rater1']
skin2 = _df['nExp']

# Average skin tone; use mean of available ratings
skin_avg = pd.concat([skin1, skin2], axis=1).mean(axis=1)

# Red cards count (rare 0-2) per dyad
red_cards = _df['yellowCards']

# Games (exposure) per dyad
games = _df['redCards']

# Basic missingness
print('skin1 missing', skin1.isna().mean())
print('skin2 missing', skin2.isna().mean())
print('skin_avg missing', skin_avg.isna().mean())

# Drop rows with missing essential values or non-positive games
mask = (~skin_avg.isna()) & (~red_cards.isna()) & (~games.isna()) & (games > 0)
_df2 = _df.loc[mask].copy()
_df2['skin_avg'] = skin_avg.loc[mask]
_df2['red_cards'] = red_cards.loc[mask]
_df2['games'] = games.loc[mask]

# Define dark vs light based on midpoint 0.5
_df2['skin_group'] = np.where(_df2['skin_avg'] > 0.5, 'dark',
                             np.where(_df2['skin_avg'] < 0.5, 'light', 'mid'))

print('group counts', _df2['skin_group'].value_counts())

# Compute red card rates per game by group
_df2['rate'] = _df2['red_cards'] / _df2['games']

rate_stats = _df2.groupby('skin_group')['rate'].agg(['mean','median','count'])
print('rate stats by group')
print(rate_stats)

# Compare dark vs light (exclude mid)
_df_dl = _df2[_df2['skin_group'].isin(['dark','light'])].copy()

# Poisson regression with offset for games
X = sm.add_constant(_df_dl['skin_avg'])
model = sm.GLM(_df_dl['red_cards'], X, family=sm.families.Poisson(), offset=np.log(_df_dl['games']))
res = model.fit(cov_type='HC0')
print(res.summary())

# Effect size: rate ratio for 0.25 increase and for dark vs light means
beta = res.params['skin_avg']
se = res.bse['skin_avg']
pval = res.pvalues['skin_avg']

# compute rate ratio for 0.5 change (light=0.25, dark=0.75 for example)
rr_05 = np.exp(beta * 0.5)

# predicted rate for light vs dark using group means
mean_light = _df_dl.loc[_df_dl['skin_group']=='light','skin_avg'].mean()
mean_dark = _df_dl.loc[_df_dl['skin_group']=='dark','skin_avg'].mean()
rr_group = np.exp(beta * (mean_dark - mean_light))

print('beta', beta, 'se', se, 'p', pval)
print('rr_0.5', rr_05)
print('mean_light', mean_light, 'mean_dark', mean_dark, 'rr_group', rr_group)

# Also compute simple difference in rates with t-test (approx)
from scipy import stats
light_rates = _df_dl.loc[_df_dl['skin_group']=='light','rate']
dark_rates = _df_dl.loc[_df_dl['skin_group']=='dark','rate']

# Use Welch t-test
stat, pval_t = stats.ttest_ind(dark_rates, light_rates, equal_var=False)
print('t-test p', pval_t, 'dark mean', dark_rates.mean(), 'light mean', light_rates.mean())

# Save key metrics to file for later use
metrics = {
    'n_total': int(len(_df2)),
    'n_dark': int((_df2['skin_group']=='dark').sum()),
    'n_light': int((_df2['skin_group']=='light').sum()),
    'n_mid': int((_df2['skin_group']=='mid').sum()),
    'rate_mean_dark': float(dark_rates.mean()),
    'rate_mean_light': float(light_rates.mean()),
    'rate_median_dark': float(dark_rates.median()),
    'rate_median_light': float(light_rates.median()),
    'poisson_beta': float(beta),
    'poisson_se': float(se),
    'poisson_p': float(pval),
    'rr_0.5': float(rr_05),
    'rr_group': float(rr_group),
}

import json
with open('metrics.json','w') as f:
    json.dump(metrics, f, indent=2)
