import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Load data
df = pd.read_csv('mortgage.csv')

# Variables
gender = df['feature2']  # 1 female, 0 male
approved = df['feature14']  # 1 accepted, 0 denied
denied = df['feature11']

# Basic sanity: approved should be 1 - denied?
mismatch = (approved + denied != 1).sum()

# Clean for analysis
df = df.replace([np.inf, -np.inf], np.nan)

# Group approval rates (drop rows with missing gender/approval)
df_basic = df[['feature2', 'feature14']].dropna()
rates = df_basic.groupby('feature2')['feature14'].mean()
counts = df_basic.groupby('feature2')['feature14'].agg(['mean', 'count', 'sum'])

# Two-proportion z-test
# female =1, male=0
n_f = counts.loc[1, 'count']
n_m = counts.loc[0, 'count']
p_f = counts.loc[1, 'mean']
p_m = counts.loc[0, 'mean']

# pooled proportion
p_pool = (counts.loc[1, 'sum'] + counts.loc[0, 'sum']) / (n_f + n_m)
se = (p_pool * (1 - p_pool) * (1 / n_f + 1 / n_m)) ** 0.5
z = (p_f - p_m) / se if se > 0 else float('nan')
pval_z = 2 * stats.norm.sf(abs(z)) if se > 0 else float('nan')

# Unadjusted logistic regression
df_unadj = df[['feature2', 'feature14']].dropna()
X = sm.add_constant(df_unadj['feature2'])
logit_unadj = sm.Logit(df_unadj['feature14'], X).fit(disp=False)

# Adjusted logistic regression with other features (exclude outcome columns)
# Use all other features except feature11 and feature14
predictors = df.drop(columns=['feature11', 'feature14'])
df_adj = pd.concat([predictors, df['feature14']], axis=1).dropna()
X_adj = sm.add_constant(df_adj.drop(columns=['feature14']))
logit_adj = sm.Logit(df_adj['feature14'], X_adj).fit(disp=False)

# Extract gender coefficient and odds ratio
coef_unadj = logit_unadj.params['feature2']
p_unadj = logit_unadj.pvalues['feature2']
or_unadj = float(np.exp(coef_unadj))

coef_adj = logit_adj.params['feature2']
p_adj = logit_adj.pvalues['feature2']
or_adj = float(np.exp(coef_adj))

# Effect size: difference in approval rates
diff = p_f - p_m

print('mismatch_approved_denied', mismatch)
print('counts', counts.to_dict())
print('approval_rate_female', p_f, 'male', p_m, 'diff', diff)
print('z_test', z, 'pval', pval_z)
print('logit_unadj coef', coef_unadj, 'or', or_unadj, 'p', p_unadj)
print('logit_adj coef', coef_adj, 'or', or_adj, 'p', p_adj)
