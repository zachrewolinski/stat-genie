import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data

df = pd.read_csv('crofoot.csv')

# Map variables based on info.json descriptions
# outcome: m_focal (1 if focal wins)
# focal group size: f_other
# other group size: win
# focal distance from center: m_other
# other distance from center: n_focal

outcome = df['m_focal']
size_focal = df['f_other']
size_other = df['win']

dist_focal = df['m_other']
dist_other = df['n_focal']

# Relative measures
rel_size = size_focal - size_other
rel_dist = dist_focal - dist_other

# Standardize predictors for stability
X = pd.DataFrame({
    'rel_size': (rel_size - rel_size.mean())/rel_size.std(),
    'rel_dist': (rel_dist - rel_dist.mean())/rel_dist.std(),
})
X = sm.add_constant(X)

model = sm.Logit(outcome, X)
res = model.fit(disp=False)
print(res.summary())

# Also check bivariate models for each predictor
for name in ['rel_size', 'rel_dist']:
    X1 = sm.add_constant(X[[name]])
    res1 = sm.Logit(outcome, X1).fit(disp=False)
    print('\nBivariate:', name)
    print(res1.summary())

# Compute simple odds ratios for 1 SD change
params = res.params
conf = res.conf_int()
print('\nOdds ratios (1 SD):')
for name in ['rel_size', 'rel_dist']:
    or_ = np.exp(params[name])
    ci_low, ci_high = np.exp(conf.loc[name])
    print(name, or_, ci_low, ci_high)

