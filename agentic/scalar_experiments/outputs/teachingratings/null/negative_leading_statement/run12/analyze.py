import json
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

DATA_PATH = 'teachingratings.csv'

df = pd.read_csv(DATA_PATH)

# Basic correlation
corr = df['beauty'].corr(df['eval'])

# Simple OLS
model_simple = smf.ols('eval ~ beauty', data=df).fit()

# OLS with controls (categorical as C())
formula = (
    'eval ~ beauty + age + students + allstudents '
    '+ C(minority) + C(gender) + C(credits) + C(division) + C(native) + C(tenure)'
)
model_controls = smf.ols(formula, data=df).fit()

# Standardized effect for beauty in controlled model
beauty_std = df['beauty'].std(ddof=1)
eval_std = df['eval'].std(ddof=1)
coef = model_controls.params['beauty']
std_effect = coef * (beauty_std / eval_std)

results = {
    'n': int(len(df)),
    'corr_beauty_eval': corr,
    'simple_coef': model_simple.params['beauty'],
    'simple_p': model_simple.pvalues['beauty'],
    'controls_coef': coef,
    'controls_p': model_controls.pvalues['beauty'],
    'controls_ci': list(model_controls.conf_int().loc['beauty'].values),
    'std_effect': std_effect,
    'r2_simple': model_simple.rsquared,
    'r2_controls': model_controls.rsquared,
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)
