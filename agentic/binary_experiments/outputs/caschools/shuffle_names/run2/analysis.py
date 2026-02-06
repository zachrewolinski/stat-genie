import pandas as pd
import statsmodels.api as sm

# Load data
_df = pd.read_csv('caschools.csv')

# Based on metadata: english = total enrollment, students = number of teachers
_df['student_teacher_ratio'] = _df['english'] / _df['students']

# Based on metadata: district = reading score, expenditure = math score
_df['avg_score'] = (_df['district'] + _df['expenditure']) / 2

# Correlation between ratio and performance
corr = _df['student_teacher_ratio'].corr(_df['avg_score'])

# Simple linear regression
X = sm.add_constant(_df['student_teacher_ratio'])
model = sm.OLS(_df['avg_score'], X).fit()

print('Student-teacher ratio vs avg score correlation:', corr)
print('\nRegression summary (avg_score ~ student_teacher_ratio):')
print(model.summary())
