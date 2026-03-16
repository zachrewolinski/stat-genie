import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
df = pd.read_csv('crofoot.csv')

# Derived variables
df['rel_size'] = df['n_focal'] - df['n_other']
df['loc_adv'] = df['dist_other'] - df['dist_focal']

X = sm.add_constant(df[['rel_size', 'loc_adv']])
model = sm.Logit(df['win'], X).fit(disp=False)

params = model.params
pvals = model.pvalues
conf = model.conf_int()

# Odds ratios
odds = np.exp(params)
conf_or = np.exp(conf)

print('N', len(df))
print('Model LLR p-value', model.llr_pvalue)
print('Pseudo R2', model.prsquared)
print('rel_size coef', params['rel_size'], 'p', pvals['rel_size'], 'OR', odds['rel_size'], 'CI', tuple(conf_or.loc['rel_size']))
print('loc_adv coef', params['loc_adv'], 'p', pvals['loc_adv'], 'OR per meter', odds['loc_adv'], 'CI', tuple(conf_or.loc['loc_adv']))

# Scale loc_adv to 100m for interpretability
coef_100 = params['loc_adv'] * 100
or_100 = np.exp(coef_100)
conf_or_100 = np.exp(conf.loc['loc_adv'] * 100)
print('loc_adv OR per 100m', or_100, 'CI', tuple(conf_or_100))
