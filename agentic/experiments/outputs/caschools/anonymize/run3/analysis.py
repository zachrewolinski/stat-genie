import pandas as pd
import statsmodels.api as sm

# Load data
_df = pd.read_csv('caschools.csv')

# Student-teacher ratio
_df['stratio'] = _df['feature6'] / _df['feature7']

# Academic performance: average of reading and math
_df['avg_score'] = _df[['feature14', 'feature15']].mean(axis=1)

# Correlation
corr = _df['stratio'].corr(_df['avg_score'])

# Simple OLS
X = sm.add_constant(_df['stratio'])
model = sm.OLS(_df['avg_score'], X).fit()

# OLS with controls (income, percent English learners, lunch)
controls = _df[['stratio', 'feature12', 'feature13', 'feature9']].copy()
controls = sm.add_constant(controls)
model_ctrl = sm.OLS(_df['avg_score'], controls).fit()

print('Correlation between student-teacher ratio and avg_score:', corr)
print('\nSimple OLS:\n', model.summary())
print('\nOLS with controls (income, English learners, lunch):\n', model_ctrl.summary())
