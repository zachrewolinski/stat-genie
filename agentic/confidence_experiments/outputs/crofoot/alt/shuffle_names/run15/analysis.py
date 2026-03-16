import pandas as pd
import numpy as np
import statsmodels.api as sm


df = pd.read_csv('crofoot.csv')

# Identify focal and other group sizes based on deterministic mapping from IDs
# f_other = dist_focal + other (focal total)
# win = focal + f_focal (other total)

focal_total = df['f_other']
other_total = df['win']

# Distances from home-range centers (two columns in meters)
# Column names are shuffled, but both are distances for the two groups
# We'll treat m_other as focal distance and n_focal as other distance; sign of relative
# location would flip if swapped, not affecting significance.

dist_focal = df['m_other']
dist_other = df['n_focal']

# Binary outcome: focal group win
win = df['m_focal']

# Predictors
rel_size = focal_total - other_total
rel_loc = dist_focal - dist_other

# Standardize for effect size comparability
rel_size_z = (rel_size - rel_size.mean()) / rel_size.std(ddof=0)
rel_loc_z = (rel_loc - rel_loc.mean()) / rel_loc.std(ddof=0)

X = pd.DataFrame({
    'rel_size_z': rel_size_z,
    'rel_loc_z': rel_loc_z,
})
X = sm.add_constant(X)

model = sm.Logit(win, X)
result = model.fit(disp=False)

# Also run univariate models for each predictor
X_size = sm.add_constant(pd.DataFrame({'rel_size_z': rel_size_z}))
res_size = sm.Logit(win, X_size).fit(disp=False)

X_loc = sm.add_constant(pd.DataFrame({'rel_loc_z': rel_loc_z}))
res_loc = sm.Logit(win, X_loc).fit(disp=False)

print('Multivariate model')
print(result.summary())
print('\nUnivariate: rel_size')
print(res_size.summary())
print('\nUnivariate: rel_loc')
print(res_loc.summary())

# Save key stats for downstream use
out = {
    'n': int(len(df)),
    'rel_size_coef': float(result.params['rel_size_z']),
    'rel_size_p': float(result.pvalues['rel_size_z']),
    'rel_loc_coef': float(result.params['rel_loc_z']),
    'rel_loc_p': float(result.pvalues['rel_loc_z']),
    'rel_size_uni_p': float(res_size.pvalues['rel_size_z']),
    'rel_loc_uni_p': float(res_loc.pvalues['rel_loc_z']),
}

pd.Series(out).to_json('analysis_results.json')
