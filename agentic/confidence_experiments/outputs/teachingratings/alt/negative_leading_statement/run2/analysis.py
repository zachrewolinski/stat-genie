import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
path = 'teachingratings.csv'
df = pd.read_csv(path)

# Basic stats
n = len(df)

# Pearson correlation
corr, corr_p = stats.pearsonr(df['beauty'], df['eval'])

# Simple regression
model_simple = smf.ols('eval ~ beauty', data=df).fit(cov_type='HC3')

# Multiple regression with controls
formula = (
    'eval ~ beauty + age + C(gender) + C(minority) + C(credits) + '
    'C(division) + C(native) + C(tenure) + students + allstudents'
)
model_controls = smf.ols(formula, data=df).fit(cov_type='HC3')

# Standardized effect of beauty in controls model
# Standardize beauty and eval
beauty_z = (df['beauty'] - df['beauty'].mean()) / df['beauty'].std(ddof=0)
eval_z = (df['eval'] - df['eval'].mean()) / df['eval'].std(ddof=0)
df_z = df.copy()
df_z['beauty_z'] = beauty_z
df_z['eval_z'] = eval_z
model_controls_z = smf.ols(
    'eval_z ~ beauty_z + age + C(gender) + C(minority) + C(credits) + '
    'C(division) + C(native) + C(tenure) + students + allstudents',
    data=df_z
).fit(cov_type='HC3')

# Summaries
results = {
    'n': n,
    'corr': corr,
    'corr_p': corr_p,
    'simple_coef': model_simple.params['beauty'],
    'simple_p': model_simple.pvalues['beauty'],
    'simple_ci': model_simple.conf_int().loc['beauty'].tolist(),
    'controls_coef': model_controls.params['beauty'],
    'controls_p': model_controls.pvalues['beauty'],
    'controls_ci': model_controls.conf_int().loc['beauty'].tolist(),
    'controls_r2': model_controls.rsquared,
    'std_coef': model_controls_z.params['beauty_z'],
    'std_p': model_controls_z.pvalues['beauty_z'],
    'eval_sd': df['eval'].std(ddof=0),
    'beauty_sd': df['beauty'].std(ddof=0),
}

print(results)
