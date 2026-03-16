import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'crofoot.csv'
df = pd.read_csv(path)

# Map columns using descriptions from info.json (shuffled names)
# Outcome: 1 if focal won contest, 0 if other won
outcome_col = 'm_focal'
# Group sizes
focal_size_col = 'f_other'  # Number of individuals in focal group
other_size_col = 'win'      # Number of individuals in other group
# Distances from home range centers
focal_dist_col = 'm_other'  # Distance of focal group from center of its home range
other_dist_col = 'n_focal'  # Distance of other group from center of its home range

# Construct predictors
rel_size = df[focal_size_col] - df[other_size_col]
loc_adv = df[other_dist_col] - df[focal_dist_col]  # positive => contest closer to focal center

# Standardize predictors for effect size comparability
rel_size_z = (rel_size - rel_size.mean()) / rel_size.std(ddof=0)
loc_adv_z = (loc_adv - loc_adv.mean()) / loc_adv.std(ddof=0)

# Prepare design matrix
X = pd.DataFrame({
    'rel_size_z': rel_size_z,
    'loc_adv_z': loc_adv_z
})
X = sm.add_constant(X)
y = df[outcome_col]

# Logistic regression
model = sm.Logit(y, X)
result = model.fit(disp=False)

# Also fit univariate models for clarity
X_size = sm.add_constant(pd.DataFrame({'rel_size_z': rel_size_z}))
res_size = sm.Logit(y, X_size).fit(disp=False)

X_loc = sm.add_constant(pd.DataFrame({'loc_adv_z': loc_adv_z}))
res_loc = sm.Logit(y, X_loc).fit(disp=False)

# Compute odds ratios and CI
params = result.params
conf = result.conf_int()
odds = np.exp(params)
conf_odds = np.exp(conf)

# Summaries
print('Multivariate logit:')
print(result.summary())
print('\nOdds ratios (multivariate):')
print(pd.DataFrame({'OR': odds, 'CI_low': conf_odds[0], 'CI_high': conf_odds[1]}))

print('\nUnivariate logit (relative size):')
print(res_size.summary())

print('\nUnivariate logit (location advantage):')
print(res_loc.summary())

# Simple descriptive stats
print('\nDescriptives:')
print(pd.DataFrame({
    'rel_size': rel_size,
    'loc_adv': loc_adv,
    'win': y
}).describe())

# Correlations with outcome
print('\nPoint-biserial correlations:')
print(pd.DataFrame({
    'rel_size_z': rel_size_z,
    'loc_adv_z': loc_adv_z,
    'win': y
}).corr())
