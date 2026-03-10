import pandas as pd
import statsmodels.api as sm
from scipy import stats


df = pd.read_csv('teachingratings.csv')

# Ensure numeric columns are numeric
for col in ['beauty', 'allstudents']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Basic correlation
valid = df[['beauty', 'allstudents']].dropna()
pearson_r, pearson_p = stats.pearsonr(valid['beauty'], valid['allstudents'])

# Simple OLS
X = sm.add_constant(valid['beauty'])
model = sm.OLS(valid['allstudents'], X).fit()

# Also compute standardized effect (beta) by z-scoring
z = (valid - valid.mean()) / valid.std(ddof=0)
Xz = sm.add_constant(z['beauty'])
model_z = sm.OLS(z['allstudents'], Xz).fit()

print('n', len(valid))
print('pearson_r', pearson_r)
print('pearson_p', pearson_p)
print('ols_coef', model.params['beauty'])
print('ols_p', model.pvalues['beauty'])
print('ols_r2', model.rsquared)
print('ols_ci', model.conf_int().loc['beauty'].tolist())
print('std_beta', model_z.params['beauty'])
print('std_p', model_z.pvalues['beauty'])
