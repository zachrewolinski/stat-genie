import pandas as pd

df = pd.read_csv('soccer.csv')

r1_var = df.groupby('refNum')['rater1'].nunique(dropna=True)
r2_var = df.groupby('refNum')['rater2'].nunique(dropna=True)

print('referees with >1 unique rater1:', (r1_var > 1).sum())
print('referees with >1 unique rater2:', (r2_var > 1).sum())

print('rater1 unique counts summary:', r1_var.describe())
