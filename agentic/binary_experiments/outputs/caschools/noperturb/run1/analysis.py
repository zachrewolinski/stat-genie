import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('caschools.csv')

# Student-teacher ratio
_df = df.copy()
_df['stratio'] = _df['students'] / _df['teachers']
_df['avg_score'] = (_df['read'] + _df['math']) / 2

# Simple correlation
corr = _df[['stratio', 'avg_score']].corr().iloc[0, 1]

# OLS regression: avg_score ~ stratio
X = sm.add_constant(_df['stratio'])
model = sm.OLS(_df['avg_score'], X).fit()

# Also report separate read and math regressions for context
model_read = sm.OLS(_df['read'], X).fit()
model_math = sm.OLS(_df['math'], X).fit()

print('Student-teacher ratio summary:')
print(_df['stratio'].describe())
print('\nAverage score summary:')
print(_df['avg_score'].describe())
print(f"\nCorrelation (stratio vs avg_score): {corr:.4f}")

print('\nOLS avg_score ~ stratio')
print(model.summary())

print('\nOLS read ~ stratio')
print(model_read.summary())

print('\nOLS math ~ stratio')
print(model_math.summary())
