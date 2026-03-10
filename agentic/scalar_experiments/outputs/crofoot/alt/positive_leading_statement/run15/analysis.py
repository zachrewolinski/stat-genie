import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
df = pd.read_csv('crofoot.csv')

# Create variables
df['size_diff'] = df['n_focal'] - df['n_other']
df['size_ratio'] = df['n_focal'] / df['n_other']
df['dist_diff'] = df['dist_other'] - df['dist_focal']  # positive means focal closer to its center
df['dist_diff_100'] = df['dist_diff'] / 100.0

# Logistic regression with size_diff and dist_diff
X = df[['size_diff', 'dist_diff_100']]
X = sm.add_constant(X)
y = df['win']

model = sm.Logit(y, X)
result = model.fit(disp=False, cov_type='HC1')

# Also model with size_ratio for robustness
Xr = df[['size_ratio', 'dist_diff_100']]
Xr = sm.add_constant(Xr)
model_r = sm.Logit(y, Xr)
result_r = model_r.fit(disp=False, cov_type='HC1')

# Bivariate models
X_size = sm.add_constant(df[['size_diff']])
res_size = sm.Logit(y, X_size).fit(disp=False, cov_type='HC1')

X_dist = sm.add_constant(df[['dist_diff_100']])
res_dist = sm.Logit(y, X_dist).fit(disp=False, cov_type='HC1')

# Summaries
print('N:', len(df))
print('\nMain model (size_diff, dist_diff_100) with robust SE:')
print(pd.DataFrame({
    'coef': result.params,
    'se': result.bse,
    'p': result.pvalues
}))

print('\nMain model (size_ratio, dist_diff_100) with robust SE:')
print(pd.DataFrame({
    'coef': result_r.params,
    'se': result_r.bse,
    'p': result_r.pvalues
}))

print('\nBivariate size_diff model:')
print(pd.DataFrame({'coef': res_size.params, 'se': res_size.bse, 'p': res_size.pvalues}))

print('\nBivariate dist_diff_100 model:')
print(pd.DataFrame({'coef': res_dist.params, 'se': res_dist.bse, 'p': res_dist.pvalues}))

# Effect size: odds ratio for 1-unit size_diff and 100m dist difference
or_size = np.exp(result.params['size_diff'])
or_dist = np.exp(result.params['dist_diff_100'])
print('\nOdds ratios (main model):')
print('OR size_diff (per 1 individual):', or_size)
print('OR dist_diff_100 (per 100m):', or_dist)

# Pseudo R2
print('\nPseudo R2 (McFadden):', result.prsquared)
