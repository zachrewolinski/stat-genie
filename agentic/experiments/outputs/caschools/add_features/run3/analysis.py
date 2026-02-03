import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('caschools.csv')

# Compute student-teacher ratio
# Avoid division by zero just in case

df = df.copy()
df['str'] = df['students'] / df['teachers']

# Academic performance: average of reading and math scores
if 'read' in df.columns and 'math' in df.columns:
    df['avg_score'] = (df['read'] + df['math']) / 2
else:
    raise ValueError('Expected read and math columns in dataset')

# Drop rows with missing values in key fields
analysis_df = df[['avg_score', 'str']].dropna()

# Correlation
corr = analysis_df['avg_score'].corr(analysis_df['str'])

# OLS regression avg_score ~ str
X = sm.add_constant(analysis_df['str'])
model = sm.OLS(analysis_df['avg_score'], X).fit()

print('n:', len(analysis_df))
print('corr(avg_score, str):', corr)
print(model.summary())
