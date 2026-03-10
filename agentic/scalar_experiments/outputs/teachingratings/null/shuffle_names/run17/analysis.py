import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
path = 'teachingratings.csv'
df = pd.read_csv(path)

# Basic correlation
corr = df['beauty'].corr(df['allstudents'])

# OLS without controls
model_simple = smf.ols('allstudents ~ beauty', data=df).fit()

# OLS with controls
# Treat categorical variables as categories
cat_cols = ['eval','tenure','prof','native','gender','credits']
for c in cat_cols:
    df[c] = df[c].astype('category')

model_controls = smf.ols('allstudents ~ beauty + age + C(tenure) + C(prof) + C(native) + C(gender) + C(credits) + rownames + minority + students', data=df).fit()

# Extract key stats
simple_coef = model_simple.params['beauty']
simple_p = model_simple.pvalues['beauty']

ctrl_coef = model_controls.params['beauty']
ctrl_p = model_controls.pvalues['beauty']

# Print summary stats for inspection
print('corr', corr)
print('simple_coef', simple_coef, 'simple_p', simple_p, 'r2', model_simple.rsquared)
print('ctrl_coef', ctrl_coef, 'ctrl_p', ctrl_p, 'r2', model_controls.rsquared)

# Additional: effect in SDs of eval per SD beauty? compute standardized coef
beauty_std = df['beauty'].std()
eval_std = df['allstudents'].std()
std_beta = simple_coef * beauty_std / eval_std
print('std_beta_simple', std_beta)

# Save for later if needed

