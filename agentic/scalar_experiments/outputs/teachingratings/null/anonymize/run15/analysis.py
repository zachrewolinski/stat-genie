import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm


df = pd.read_csv('teachingratings.csv')

# Map categorical variables for modeling; keep as strings for patsy C()

# Simple correlation
corr = df['feature6'].corr(df['feature7'])
print('corr', corr)

# Simple OLS
m_simple = smf.ols('feature7 ~ feature6', data=df).fit(cov_type='HC3')
print('simple coef', m_simple.params['feature6'], 'p', m_simple.pvalues['feature6'])
print('simple r2', m_simple.rsquared)

# Multivariate model with controls
formula = (
    'feature7 ~ feature6 + feature3 + C(feature4) + C(feature2) + '
    'C(feature5) + C(feature8) + C(feature9) + C(feature10) + '
    'feature11 + feature12'
)

m_full = smf.ols(formula, data=df).fit(cov_type='HC3')
print('full coef', m_full.params['feature6'], 'p', m_full.pvalues['feature6'])
print('full r2', m_full.rsquared)

# Standardized effect for beauty
# Standardize feature6 and feature7 to compute standardized beta
zdf = df.copy()
for col in ['feature6','feature7','feature3','feature11','feature12']:
    zdf[col] = (zdf[col] - zdf[col].mean()) / zdf[col].std(ddof=0)

m_full_std = smf.ols(
    'feature7 ~ feature6 + feature3 + C(feature4) + C(feature2) + '
    'C(feature5) + C(feature8) + C(feature9) + C(feature10) + '
    'feature11 + feature12',
    data=zdf
).fit(cov_type='HC3')
print('full std beta', m_full_std.params['feature6'], 'p', m_full_std.pvalues['feature6'])

# Model with instructor fixed effects? There is feature13 instructor id; might be multiple courses per instructor.
# If we include instructor fixed effects, beauty likely constant per instructor. We can use cluster-robust SE by instructor.
# Let's compute clustered SE by instructor for the full model.

m_full_cluster = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['feature13']})
print('full cluster coef', m_full_cluster.params['feature6'], 'p', m_full_cluster.pvalues['feature6'])

# Save key stats to a small dict for later use
print('n', len(df))
