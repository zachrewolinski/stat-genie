import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


df = pd.read_csv('teachingratings.csv')

# Basic cleaning: ensure numeric types where expected
numeric_cols = ['beauty', 'allstudents', 'age', 'division', 'rownames', 'minority', 'students']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing key vars
key_df = df[['beauty', 'allstudents']].dropna()

# Correlation and simple OLS
corr = key_df['beauty'].corr(key_df['allstudents'])

simple_model = smf.ols('allstudents ~ beauty', data=df).fit()

# Controlled model: include other instructor/course characteristics.
# Categorical vars: eval, tenure, prof, native, gender, credits
# Include age, students, minority, rownames (counts) as numeric covariates.
control_formula = (
    'allstudents ~ beauty + age + students + minority + rownames + C(eval) + C(tenure) + '
    'C(prof) + C(native) + C(gender) + C(credits)'
)
controlled_model = smf.ols(control_formula, data=df).fit()

# Compute standardized effect of beauty in simple model (beta * SD_x / SD_y)
beauty_sd = key_df['beauty'].std(ddof=1)
allstudents_sd = key_df['allstudents'].std(ddof=1)
std_effect_simple = simple_model.params['beauty'] * beauty_sd / allstudents_sd

# 95% CI for beauty coefficient in controlled model
conf_int = controlled_model.conf_int().loc['beauty'].tolist()

results = {
    'n': int(len(df)),
    'corr': corr,
    'simple_coef': simple_model.params['beauty'],
    'simple_p': simple_model.pvalues['beauty'],
    'simple_r2': simple_model.rsquared,
    'std_effect_simple': std_effect_simple,
    'controlled_coef': controlled_model.params['beauty'],
    'controlled_p': controlled_model.pvalues['beauty'],
    'controlled_r2': controlled_model.rsquared,
    'controlled_ci_low': conf_int[0],
    'controlled_ci_high': conf_int[1],
}

print(json.dumps(results, indent=2))
