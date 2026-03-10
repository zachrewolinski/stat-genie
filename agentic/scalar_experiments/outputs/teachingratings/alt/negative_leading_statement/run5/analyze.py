import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'teachingratings.csv'
df = pd.read_csv(path)

# Basic info
print('rows', len(df))

# Ensure numeric types
# beauty and eval are numeric already

# Simple correlation
corr = df['beauty'].corr(df['eval'])
print('corr_beauty_eval', corr)

# Simple OLS
model_simple = smf.ols('eval ~ beauty', data=df).fit()
print('simple_coef', model_simple.params['beauty'])
print('simple_p', model_simple.pvalues['beauty'])
print('simple_r2', model_simple.rsquared)

# OLS with controls
# Use categorical for minority, gender, credits, division, native, tenure
# Use age, students, allstudents as numeric controls
formula = 'eval ~ beauty + age + students + allstudents + C(minority) + C(gender) + C(credits) + C(division) + C(native) + C(tenure)'
model_ctrl = smf.ols(formula, data=df).fit()
print('ctrl_coef', model_ctrl.params['beauty'])
print('ctrl_p', model_ctrl.pvalues['beauty'])
print('ctrl_r2', model_ctrl.rsquared)

# Robust SE (HC3) for sensitivity
model_ctrl_robust = model_ctrl.get_robustcov_results(cov_type='HC3')
print('ctrl_coef_robust', model_ctrl_robust.params[model_ctrl_robust.model.exog_names.index('beauty')])
print('ctrl_p_robust', model_ctrl_robust.pvalues[model_ctrl_robust.model.exog_names.index('beauty')])

# Standardize variables for effect size (beta)
# zscore beauty and eval then regress
zdf = df.copy()
for col in ['beauty', 'eval', 'age', 'students', 'allstudents']:
    zdf[col] = (zdf[col] - zdf[col].mean()) / zdf[col].std(ddof=0)

model_beta = smf.ols('eval ~ beauty + age + students + allstudents + C(minority) + C(gender) + C(credits) + C(division) + C(native) + C(tenure)', data=zdf).fit()
print('beta_coef', model_beta.params['beauty'])
print('beta_p', model_beta.pvalues['beauty'])

# Save summary for later
model_simple_summary = model_simple.summary().as_text()
model_ctrl_summary = model_ctrl.summary().as_text()

with open('analysis_output.txt', 'w') as f:
    f.write('corr_beauty_eval: %s\n' % corr)
    f.write('simple_coef: %s\n' % model_simple.params['beauty'])
    f.write('simple_p: %s\n' % model_simple.pvalues['beauty'])
    f.write('simple_r2: %s\n' % model_simple.rsquared)
    f.write('ctrl_coef: %s\n' % model_ctrl.params['beauty'])
    f.write('ctrl_p: %s\n' % model_ctrl.pvalues['beauty'])
    f.write('ctrl_r2: %s\n' % model_ctrl.rsquared)
    f.write('ctrl_coef_robust: %s\n' % model_ctrl_robust.params[model_ctrl_robust.model.exog_names.index('beauty')])
    f.write('ctrl_p_robust: %s\n' % model_ctrl_robust.pvalues[model_ctrl_robust.model.exog_names.index('beauty')])
    f.write('beta_coef: %s\n' % model_beta.params['beauty'])
    f.write('beta_p: %s\n' % model_beta.pvalues['beauty'])

