import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
from scipy import stats

# Load data

df = pd.read_csv('teachingratings.csv')

print('rows', len(df))
print(df.head())
print(df.dtypes)

# Basic correlation between beauty and eval
corr, pval = stats.pearsonr(df['beauty'], df['eval'])
print('pearson_corr', corr, 'p', pval)

# Simple OLS
model_simple = smf.ols('eval ~ beauty', data=df).fit()
print(model_simple.summary())

# Add controls
# Categorical variables: minority, gender, credits, division, native, tenure
# Numeric controls: age, students, allstudents
# Potentially log students? We'll use log(allstudents) and log(students) to handle skew.

df['log_students'] = np.log(df['students'])
df['log_allstudents'] = np.log(df['allstudents'])

model_controls = smf.ols(
    'eval ~ beauty + age + log_students + log_allstudents + C(minority) + C(gender) + C(credits) + C(division) + C(native) + C(tenure)',
    data=df
).fit()
print(model_controls.summary())

# Alternative: use only one size variable to avoid collinearity
model_controls2 = smf.ols(
    'eval ~ beauty + age + log_students + C(minority) + C(gender) + C(credits) + C(division) + C(native) + C(tenure)',
    data=df
).fit()
print(model_controls2.summary())

# Standardized effect size for beauty in simple model
beauty_std = df['beauty'].std()
eval_std = df['eval'].std()
coef_simple = model_simple.params['beauty']
std_effect = coef_simple * beauty_std / eval_std
print('std_effect_simple', std_effect)
