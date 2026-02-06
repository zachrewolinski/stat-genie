import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('caschools.csv')

# Student-teacher ratio
_df['str'] = _df['students'] / _df['teachers']

# Academic performance: average of reading and math
_df['score_avg'] = (_df['read'] + _df['math']) / 2.0

# Simple correlation
corr = _df['str'].corr(_df['score_avg'])

# OLS regression: score on student-teacher ratio
X = sm.add_constant(_df['str'])
model = sm.OLS(_df['score_avg'], X).fit()

print('Correlation(str, score_avg):', corr)
print(model.summary())
