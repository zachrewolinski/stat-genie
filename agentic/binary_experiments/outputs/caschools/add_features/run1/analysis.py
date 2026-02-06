import pandas as pd
import statsmodels.api as sm

# Load data

df = pd.read_csv('caschools.csv')

# Create student-teacher ratio and academic performance (average of read and math)

df['str'] = df['students'] / df['teachers']
df['score'] = (df['read'] + df['math']) / 2

# Simple linear regression: score ~ str
X = sm.add_constant(df['str'])
model = sm.OLS(df['score'], X).fit()

print(model.summary())

# Also compute correlation
corr = df['str'].corr(df['score'])
print(f"Correlation between student-teacher ratio and score: {corr:.4f}")
