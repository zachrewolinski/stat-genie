import pandas as pd
import numpy as np
import statsmodels.api as sm


df = pd.read_csv('crofoot.csv')

win_col = 'm_focal'
# based on value ranges: group sizes are columns with 5-13; distances are 55-853
size_focal = 'win'
size_other = 'f_other'

dist_focal = 'm_other'
dist_other = 'n_focal'

rel_size = df[size_focal] - df[size_other]
rel_loc = df[dist_other] - df[dist_focal]  # positive => closer to focal

X = pd.DataFrame({'rel_size': rel_size, 'rel_loc': rel_loc})
X = sm.add_constant(X)

y = df[win_col]
model = sm.Logit(y, X).fit(disp=False)

params = model.params
pvals = model.pvalues

# Odds ratios
or_rel_size = float(np.exp(params['rel_size']))
# per 100m difference for location
or_rel_loc_100m = float(np.exp(params['rel_loc'] * 100))

print('n:', len(df))
print('win_rate:', y.mean())
print('coef_rel_size:', params['rel_size'])
print('p_rel_size:', pvals['rel_size'])
print('OR_rel_size:', or_rel_size)
print('coef_rel_loc:', params['rel_loc'])
print('p_rel_loc:', pvals['rel_loc'])
print('OR_rel_loc_per_100m:', or_rel_loc_100m)
print('pseudo_r2:', model.prsquared)
