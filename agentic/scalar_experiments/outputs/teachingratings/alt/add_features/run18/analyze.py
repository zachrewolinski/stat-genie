import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
path = 'teachingratings.csv'
df = pd.read_csv(path)

# Keep relevant columns
# Core variables
cols = ['eval', 'beauty', 'age', 'students', 'allstudents', 'gender', 'minority',
        'native', 'tenure', 'division', 'credits']

# Only keep columns present
cols = [c for c in cols if c in df.columns]

df_sub = df[cols].copy()

# Drop missing
df_sub = df_sub.dropna()

# Encode categorical vars for controls
cat_cols = [c for c in ['gender', 'minority', 'native', 'tenure', 'division', 'credits'] if c in df_sub.columns]

# Simple correlation
corr = df_sub['beauty'].corr(df_sub['eval'])
# Pearson correlation test
corr_r, corr_p = stats.pearsonr(df_sub['beauty'], df_sub['eval'])

# Simple OLS: eval ~ beauty
X_simple = sm.add_constant(df_sub[['beauty']])
model_simple = sm.OLS(df_sub['eval'], X_simple).fit()

# Multiple OLS with controls
X = df_sub[['beauty']].copy()
if cat_cols:
    X = pd.concat([X, pd.get_dummies(df_sub[cat_cols], drop_first=True)], axis=1)

# add numeric controls
for c in ['age', 'students', 'allstudents']:
    if c in df_sub.columns:
        X[c] = df_sub[c]

X = sm.add_constant(X)
model_multi = sm.OLS(df_sub['eval'], X).fit()

# Standardized effect size for beauty in simple model (beta in SD units)
beauty_std = df_sub['beauty'].std()
eval_std = df_sub['eval'].std()
std_beta_simple = model_simple.params['beauty'] * (beauty_std / eval_std)

# Standardized effect size for beauty in multiple model
std_beta_multi = model_multi.params['beauty'] * (beauty_std / eval_std)

# Output key results
print('N', len(df_sub))
print('corr', corr, 'p', corr_p)
print('simple coef', model_simple.params['beauty'], 'p', model_simple.pvalues['beauty'], 'R2', model_simple.rsquared, 'std_beta', std_beta_simple)
print('multi coef', model_multi.params['beauty'], 'p', model_multi.pvalues['beauty'], 'R2', model_multi.rsquared, 'std_beta', std_beta_multi)
