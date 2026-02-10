import pandas as pd
import statsmodels.api as sm


df = pd.read_csv('caschools.csv')
df['stratio'] = df['students'] / df['teachers']
df['avgscore'] = df[['read', 'math']].mean(axis=1)

# Focus on realistic class sizes, e.g., 10-30 students per teacher
trim = df[(df['stratio'] >= 10) & (df['stratio'] <= 30)].copy()

print('Full sample N =', len(df))
print('Trimmed sample N =', len(trim))
print('Full corr(stratio, avgscore) =', df['stratio'].corr(df['avgscore']))
print('Trimmed corr(stratio, avgscore) =', trim['stratio'].corr(trim['avgscore']))

X_full = sm.add_constant(df['stratio'])
X_trim = sm.add_constant(trim['stratio'])

print('\nFull OLS:')
print(sm.OLS(df['avgscore'], X_full).fit().summary())

print('\nTrimmed OLS:')
print(sm.OLS(trim['avgscore'], X_trim).fit().summary())
