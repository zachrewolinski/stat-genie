import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('teachingratings.csv')

# Keep relevant columns for the research question
cols = [
    'eval', 'beauty', 'age', 'gender', 'minority', 'native', 'tenure',
    'division', 'credits', 'students', 'allstudents'
]

# Drop rows with missing values in any of these columns
analysis_df = df[cols].dropna().copy()

# Basic stats
n = len(analysis_df)

# Pearson correlation between beauty and eval
r, r_p = stats.pearsonr(analysis_df['beauty'], analysis_df['eval'])

# Simple linear regression
model_simple = smf.ols('eval ~ beauty', data=analysis_df).fit()

# Multiple regression with controls
# Use log for class size variables to reduce skew
analysis_df['log_students'] = np.log(analysis_df['students'])
analysis_df['log_allstudents'] = np.log(analysis_df['allstudents'])

model_controls = smf.ols(
    'eval ~ beauty + age + C(gender) + C(minority) + C(native) + C(tenure) + '
    'C(division) + C(credits) + log_students + log_allstudents',
    data=analysis_df
).fit()

# Extract key results
simple_coef = model_simple.params['beauty']
simple_p = model_simple.pvalues['beauty']
simple_ci = model_simple.conf_int().loc['beauty'].tolist()

controls_coef = model_controls.params['beauty']
controls_p = model_controls.pvalues['beauty']
controls_ci = model_controls.conf_int().loc['beauty'].tolist()

results = {
    'n': int(n),
    'pearson_r': float(r),
    'pearson_p': float(r_p),
    'simple_coef': float(simple_coef),
    'simple_p': float(simple_p),
    'simple_ci': [float(simple_ci[0]), float(simple_ci[1])],
    'controls_coef': float(controls_coef),
    'controls_p': float(controls_p),
    'controls_ci': [float(controls_ci[0]), float(controls_ci[1])],
    'simple_r2': float(model_simple.rsquared),
    'controls_r2': float(model_controls.rsquared),
}

print(json.dumps(results, indent=2))
