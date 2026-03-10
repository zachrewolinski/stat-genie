import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load data
path = 'teachingratings.csv'
df = pd.read_csv(path)

# Basic cleaning: ensure columns exist
# print(df.head())

# Simple correlation between beauty and eval
corr = df['beauty'].corr(df['eval'])

# Simple linear regression
model_simple = smf.ols('eval ~ beauty', data=df).fit()

# Multiple regression with controls
# Convert categorical variables: minority, gender, credits, division, native, tenure
model_controls = smf.ols('eval ~ beauty + age + C(gender) + C(minority) + C(credits) + C(division) + C(native) + C(tenure) + students + allstudents', data=df).fit()

# Also include professor fixed effects? maybe not; but we can check robust SE.

# Use robust (HC3) SE for main model
model_controls_robust = model_controls.get_robustcov_results(cov_type='HC3')

# Summaries
results = {
    'corr': corr,
    'simple_coef': model_simple.params['beauty'],
    'simple_p': model_simple.pvalues['beauty'],
    'simple_r2': model_simple.rsquared,
    'controls_coef': model_controls.params['beauty'],
    'controls_p': model_controls.pvalues['beauty'],
    'controls_r2': model_controls.rsquared,
    'controls_robust_coef': model_controls_robust.params[model_controls.model.exog_names.index('beauty')],
    'controls_robust_p': model_controls_robust.pvalues[model_controls.model.exog_names.index('beauty')],
}

print(results)

# Standardized effect (beta) for beauty in simple model
# Standardize variables
beauty_std = (df['beauty'] - df['beauty'].mean()) / df['beauty'].std()
eval_std = (df['eval'] - df['eval'].mean()) / df['eval'].std()
model_std = smf.ols('eval_std ~ beauty_std', data=pd.DataFrame({'beauty_std': beauty_std, 'eval_std': eval_std})).fit()
print({'std_beta': model_std.params['beauty_std'], 'std_p': model_std.pvalues['beauty_std']})

# Effect size: predicted change in eval for 1 SD increase in beauty
beauty_sd = df['beauty'].std()
coef = model_simple.params['beauty']
print({'beauty_sd': beauty_sd, 'eval_change_per_1sd': coef * beauty_sd})

