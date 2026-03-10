import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'crofoot.csv'
df = pd.read_csv(path)

# Keep relevant columns
cols = ['win','n_focal','n_other','dist_focal','dist_other','dyad']
sub = df[cols].copy()

# Drop rows with missing values in relevant columns
sub = sub.dropna()

# Compute relative group size and relative location
sub['size_diff'] = sub['n_focal'] - sub['n_other']
sub['size_ratio'] = sub['n_focal'] / sub['n_other']
# location advantage: positive when focal is closer to its home range center than the other group is to its own
sub['loc_diff'] = sub['dist_other'] - sub['dist_focal']

print('Rows used:', len(sub))
print(sub[['win','size_diff','size_ratio','loc_diff']].describe())

# Standardize predictors for comparable coefficients
sub['size_diff_z'] = (sub['size_diff'] - sub['size_diff'].mean()) / sub['size_diff'].std(ddof=0)
sub['loc_diff_z'] = (sub['loc_diff'] - sub['loc_diff'].mean()) / sub['loc_diff'].std(ddof=0)

# Logistic regression with both predictors
X = sm.add_constant(sub[['size_diff_z','loc_diff_z']])
model = sm.GLM(sub['win'], X, family=sm.families.Binomial())
res = model.fit()
print('\nGLM (standard errors):')
print(res.summary())

# Cluster-robust SE by dyad (if multiple rows per dyad)
try:
    res_cluster = model.fit(cov_type='cluster', cov_kwds={'groups': sub['dyad']})
    print('\nGLM (cluster-robust SE by dyad):')
    print(res_cluster.summary())
except Exception as e:
    print('Cluster-robust failed:', e)

# Separate models for each predictor
X_size = sm.add_constant(sub[['size_diff_z']])
res_size = sm.GLM(sub['win'], X_size, family=sm.families.Binomial()).fit()
print('\nGLM size only:')
print(res_size.summary())

X_loc = sm.add_constant(sub[['loc_diff_z']])
res_loc = sm.GLM(sub['win'], X_loc, family=sm.families.Binomial()).fit()
print('\nGLM location only:')
print(res_loc.summary())

# Effect sizes: predicted probability change from -1 SD to +1 SD for each predictor
mean_size = 0.0
mean_loc = 0.0

def predict_prob(size_z, loc_z):
    xb = res.params['const'] + res.params['size_diff_z']*size_z + res.params['loc_diff_z']*loc_z
    return 1/(1+np.exp(-xb))

p_low = predict_prob(-1, 0)
p_high = predict_prob(1, 0)
print('\nPredicted P(win) size_diff_z -1 to +1 SD (loc at mean):', p_low, p_high, 'delta', p_high-p_low)

p_low_loc = predict_prob(0, -1)
p_high_loc = predict_prob(0, 1)
print('Predicted P(win) loc_diff_z -1 to +1 SD (size at mean):', p_low_loc, p_high_loc, 'delta', p_high_loc-p_low_loc)
