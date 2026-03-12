import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
csv_path = 'teachingratings.csv'
df = pd.read_csv(csv_path)

# Keep relevant columns
# beauty: instructor appearance rating (standardized)
# allstudents: overall teaching evaluation score (1-5)

# Basic sanity checks
result = {}
result['n_rows'] = len(df)
result['beauty_mean'] = df['beauty'].mean()
result['beauty_std'] = df['beauty'].std()
result['eval_mean'] = df['allstudents'].mean()
result['eval_std'] = df['allstudents'].std()

# Pearson correlation
corr, pval = stats.pearsonr(df['beauty'], df['allstudents'])
result['pearson_r'] = corr
result['pearson_p'] = pval

# Simple OLS: allstudents ~ beauty
model_simple = smf.ols('allstudents ~ beauty', data=df).fit()
result['simple_coef'] = model_simple.params['beauty']
result['simple_p'] = model_simple.pvalues['beauty']
result['simple_ci_low'], result['simple_ci_high'] = model_simple.conf_int().loc['beauty']
result['simple_r2'] = model_simple.rsquared

# Multiple regression with available covariates
# Treat categorical vars as categories; numeric vars as numeric
# Use robust (HC3) SEs to be conservative.

# Identify covariates by column names except outcome and beauty
covariates = ['division', 'eval', 'age', 'tenure', 'prof', 'native', 'gender', 'credits', 'rownames', 'minority', 'students']
# Build formula with categorical encoding for object dtype or known categorical columns
cat_cols = [c for c in covariates if df[c].dtype == 'object']

# In case some numeric columns should be categorical (binary 0/1 not here), rely on dtype.
terms = []
for c in covariates:
    if c in cat_cols:
        terms.append(f'C({c})')
    else:
        terms.append(c)

formula = 'allstudents ~ beauty + ' + ' + '.join(terms)
model_full = smf.ols(formula, data=df).fit(cov_type='HC3')
result['full_coef'] = model_full.params['beauty']
result['full_p'] = model_full.pvalues['beauty']
result['full_ci_low'], result['full_ci_high'] = model_full.conf_int().loc['beauty']
result['full_r2'] = model_full.rsquared

# Save key results for review
out = pd.Series(result)
print(out.to_string())
