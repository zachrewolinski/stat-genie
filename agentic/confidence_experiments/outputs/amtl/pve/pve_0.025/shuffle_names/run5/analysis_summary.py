import pandas as pd
import statsmodels.formula.api as smf

_df = pd.read_csv('amtl.csv')
_df['is_human'] = (_df['tooth_class'] == 'Homo sapiens').astype(int)

# Simple group means
means = _df.groupby('tooth_class')['genus'].mean().sort_values(ascending=False)
print('Group means (genus):')
print(means)

# OLS with controls
model = smf.ols('genus ~ is_human + pop + stdev_age + age + C(sockets)', data=_df).fit(cov_type='HC3')
coef = model.params['is_human']
se = model.bse['is_human']
pval = model.pvalues['is_human']
print('\nHuman effect coef', coef, 'SE', se, 'p', pval)

# Standardized effect size (Cohen d) comparing human vs non-human raw
hum = _df[_df['is_human']==1]['genus']
non = _df[_df['is_human']==0]['genus']
import numpy as np
n1, n0 = len(hum), len(non)
pooled = np.sqrt(((n1-1)*hum.var() + (n0-1)*non.var())/(n1+n0-2))
d = (hum.mean() - non.mean())/pooled
print('Cohen d (raw):', d)

print('n_human', n1, 'n_non', n0)
