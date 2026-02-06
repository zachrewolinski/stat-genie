import pandas as pd
import statsmodels.api as sm

# Load data
path = 'caschools.csv'
df = pd.read_csv(path)

# Compute student-teacher ratio (students per teacher)
df['str'] = df['students'] / df['teachers']

# Academic performance measure: average of reading and math scores
df['score'] = (df['read'] + df['math']) / 2

# Basic correlation
corr = df['str'].corr(df['score'])

# Simple OLS regression: score ~ str
X = sm.add_constant(df['str'])
model = sm.OLS(df['score'], X).fit()

# Print key results
print('Correlation (str vs score):', corr)
print(model.summary())
