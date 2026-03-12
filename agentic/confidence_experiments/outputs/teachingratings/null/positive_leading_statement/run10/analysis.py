import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
path = 'teachingratings.csv'
df = pd.read_csv(path)

# Basic cleaning
# Ensure categorical columns are treated as categories
cat_cols = ['minority','gender','credits','division','native','tenure']
for col in cat_cols:
    df[col] = df[col].astype('category')

# Simple Pearson correlation
pearson_r, pearson_p = stats.pearsonr(df['beauty'], df['eval'])

# Simple OLS: eval ~ beauty
model_simple = smf.ols('eval ~ beauty', data=df).fit(cov_type='HC3')

# Multiple OLS with controls
# include instructor and course characteristics, plus class size measures
# Use log for student counts to reduce skew
for col in ['students','allstudents']:
    df[f'log_{col}'] = np.log(df[col])

formula = (
    'eval ~ beauty + age + C(gender) + C(minority) + C(credits) + '
    'C(division) + C(native) + C(tenure) + log_students + log_allstudents'
)
model_controls = smf.ols(formula, data=df).fit(cov_type='HC3')

# Extract key stats
simple_coef = model_simple.params['beauty']
simple_p = model_simple.pvalues['beauty']
simple_ci = model_simple.conf_int().loc['beauty'].tolist()

ctrl_coef = model_controls.params['beauty']
ctrl_p = model_controls.pvalues['beauty']
ctrl_ci = model_controls.conf_int().loc['beauty'].tolist()

results = {
    'n': int(df.shape[0]),
    'pearson_r': pearson_r,
    'pearson_p': pearson_p,
    'simple_coef': simple_coef,
    'simple_p': simple_p,
    'simple_ci': simple_ci,
    'ctrl_coef': ctrl_coef,
    'ctrl_p': ctrl_p,
    'ctrl_ci': ctrl_ci,
    'simple_r2': model_simple.rsquared,
    'ctrl_r2': model_controls.rsquared,
}

print(json.dumps(results, indent=2))
