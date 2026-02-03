import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('caschools.csv')

# Create student-teacher ratio
_df['str'] = _df['students'] / _df['teachers']

# Academic performance metric: average of read and math
_df['avg_score'] = _df[['read', 'math']].mean(axis=1)

# Simple correlation
corr = _df['str'].corr(_df['avg_score'])

# Simple linear regression: avg_score ~ str
X = sm.add_constant(_df['str'])
model = sm.OLS(_df['avg_score'], X).fit()

# Multiple regression controlling for key demographics
controls = _df[['str', 'income', 'english', 'lunch']].copy()
controls = sm.add_constant(controls)
model_controls = sm.OLS(_df['avg_score'], controls).fit()

# Save key results to a small report (stdout for debugging if run)
print('Correlation (str vs avg_score):', corr)
print('Simple regression coef:', model.params['str'], 'p=', model.pvalues['str'])
print('Controlled regression coef:', model_controls.params['str'], 'p=', model_controls.pvalues['str'])
