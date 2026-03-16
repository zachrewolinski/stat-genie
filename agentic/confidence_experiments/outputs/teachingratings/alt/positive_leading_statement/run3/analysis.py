import json
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load data
DF = pd.read_csv('teachingratings.csv')

# Basic cleaning: ensure categorical columns are treated as categories
cat_cols = ['minority','gender','credits','division','native','tenure']
for c in cat_cols:
    if c in DF.columns:
        DF[c] = DF[c].astype('category')

# Baseline correlation
corr = DF['beauty'].corr(DF['eval'])

# Simple OLS
model_simple = smf.ols('eval ~ beauty', data=DF).fit(cov_type='HC3')

# Controlled OLS (standard controls from paper)
# Include course/instructor characteristics and class size
model_controls = smf.ols(
    'eval ~ beauty + age + C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits) + students + allstudents',
    data=DF
).fit(cov_type='HC3')

# Effect size in SD units for eval per 1 SD beauty
beauty_sd = DF['beauty'].std()
eval_sd = DF['eval'].std()

beta_simple = model_simple.params['beauty']
beta_controls = model_controls.params['beauty']

std_effect_simple = beta_simple * beauty_sd / eval_sd
std_effect_controls = beta_controls * beauty_sd / eval_sd

# Collect results
results = {
    'n': int(DF.shape[0]),
    'corr_beauty_eval': float(corr),
    'simple_beta': float(beta_simple),
    'simple_p': float(model_simple.pvalues['beauty']),
    'simple_ci': [float(x) for x in model_simple.conf_int().loc['beauty'].tolist()],
    'controls_beta': float(beta_controls),
    'controls_p': float(model_controls.pvalues['beauty']),
    'controls_ci': [float(x) for x in model_controls.conf_int().loc['beauty'].tolist()],
    'std_effect_simple': float(std_effect_simple),
    'std_effect_controls': float(std_effect_controls),
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
