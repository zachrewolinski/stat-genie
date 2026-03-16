import pandas as pd
import numpy as np
import statsmodels.api as sm


df = pd.read_csv('crofoot.csv')

# Assume group id in n_other is focal, dist_other is other.
size_focal = df['f_other']
size_other = df['win']

# distance columns (meters)
dist_focal = df['m_other']
dist_other = df['n_focal']

# outcome
win = df['m_focal']

# relative predictors
rel_size = np.log(size_focal / size_other)
rel_loc = np.log(dist_other / dist_focal)
# rel_loc > 0 means contest is farther from focal center than other center (disadvantage)

# standardize
rel_size_z = (rel_size - rel_size.mean()) / rel_size.std()
rel_loc_z = (rel_loc - rel_loc.mean()) / rel_loc.std()

X = pd.DataFrame({
    'rel_size_z': rel_size_z,
    'rel_loc_z': rel_loc_z,
})
X = sm.add_constant(X)

model = sm.Logit(win, X).fit(disp=0)
print(model.summary())

# compute odds ratios and 95% CI
params = model.params
conf = model.conf_int()

or_df = pd.DataFrame({
    'OR': np.exp(params),
    'CI_low': np.exp(conf[0]),
    'CI_high': np.exp(conf[1]),
    'p_value': model.pvalues
})
print('\nOdds ratios:')
print(or_df)

# also check simple bivariate relations
# correlation with outcome (point-biserial)
for name, predictor in [('rel_size', rel_size), ('rel_loc', rel_loc)]:
    corr = np.corrcoef(predictor, win)[0,1]
    print(f'corr({name}, win): {corr:.3f}')

