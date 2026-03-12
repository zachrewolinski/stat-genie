import pandas as pd
import numpy as np
import statsmodels.api as sm

DF = pd.read_csv('crofoot.csv')
DF['rel_size'] = DF['n_focal'] - DF['n_other']
DF['loc_adv'] = DF['dist_other'] - DF['dist_focal']
DF['rel_size_z'] = (DF['rel_size'] - DF['rel_size'].mean()) / DF['rel_size'].std(ddof=0)
DF['loc_adv_z'] = (DF['loc_adv'] - DF['loc_adv'].mean()) / DF['loc_adv'].std(ddof=0)

X = sm.add_constant(DF[['rel_size_z', 'loc_adv_z']])
res = sm.GLM(DF['win'], X, family=sm.families.Binomial()).fit()

conf = res.conf_int()
params = res.params

# odds ratios and CI
or_vals = np.exp(params)
ci_or = np.exp(conf)

print('params', params.to_dict())
print('pvalues', res.pvalues.to_dict())
print('or', or_vals.to_dict())
print('or_ci', {col: (ci_or.loc[col,0], ci_or.loc[col,1]) for col in ci_or.index})
