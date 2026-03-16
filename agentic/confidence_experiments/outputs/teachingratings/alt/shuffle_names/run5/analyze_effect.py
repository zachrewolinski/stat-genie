import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load data
csv_path = 'teachingratings.csv'
df = pd.read_csv(csv_path)

# Basic correlation
corr = df['beauty'].corr(df['allstudents'])

# Simple OLS
model_simple = smf.ols('allstudents ~ beauty', data=df).fit(cov_type='HC3')

# Controlled OLS (exclude IDs/row index-like columns)
# division looks like row index (unique), students likely instructor ID; exclude both.
formula_ctrl = (
    'allstudents ~ beauty + age + rownames + minority '
    '+ C(tenure) + C(eval) + C(prof) + C(native) + C(gender) + C(credits)'
)
model_ctrl = smf.ols(formula_ctrl, data=df).fit(cov_type='HC3')

beauty_std = df['beauty'].std()

# Effect per 1 SD in beauty
simple_effect_1sd = model_simple.params['beauty'] * beauty_std
ctrl_effect_1sd = model_ctrl.params['beauty'] * beauty_std

# Collect key stats
results = {
    'corr_beauty_allstudents': corr,
    'simple_coef': model_simple.params['beauty'],
    'simple_se': model_simple.bse['beauty'],
    'simple_p': model_simple.pvalues['beauty'],
    'simple_r2': model_simple.rsquared,
    'simple_effect_1sd': simple_effect_1sd,
    'ctrl_coef': model_ctrl.params['beauty'],
    'ctrl_se': model_ctrl.bse['beauty'],
    'ctrl_p': model_ctrl.pvalues['beauty'],
    'ctrl_r2': model_ctrl.rsquared,
    'ctrl_effect_1sd': ctrl_effect_1sd,
    'n': len(df)
}

for k, v in results.items():
    print(f"{k}: {v}")

# Print model summary for reference
print('\nSimple model summary (beauty):')
print(model_simple.summary())
print('\nControlled model summary (beauty):')
print(model_ctrl.summary())
