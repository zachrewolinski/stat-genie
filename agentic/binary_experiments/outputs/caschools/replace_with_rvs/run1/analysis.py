import pandas as pd
import statsmodels.api as sm

# Load data
_df = pd.read_csv('caschools.csv')

# Compute student-teacher ratio
_df['student_teacher_ratio'] = _df['students'] / _df['teachers']

# Academic performance: average of reading and math scores
_df['avg_score'] = (_df['read'] + _df['math']) / 2.0

# Simple correlation
corr = _df['student_teacher_ratio'].corr(_df['avg_score'])

# OLS regression: avg_score on student_teacher_ratio
X = sm.add_constant(_df['student_teacher_ratio'])
model = sm.OLS(_df['avg_score'], X).fit()

print('Correlation (ratio vs avg_score):', corr)
print(model.summary())
