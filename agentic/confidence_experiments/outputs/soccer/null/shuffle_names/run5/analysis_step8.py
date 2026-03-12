import pandas as pd

_df = pd.read_csv('soccer.csv')

corr = _df[['rater1','nExp']].corr()
print(corr)

# compute agreement
both = _df[['rater1','nExp']].dropna()
print('n both', len(both))
print('pct equal', (both['rater1']==both['nExp']).mean())

