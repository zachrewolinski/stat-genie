import pandas as pd


df = pd.read_csv('soccer.csv')

for col in ['rater1', 'nExp']:
    vals = sorted(df[col].dropna().unique())
    print(col, vals[:10], '... total', len(vals))

print('rater2 min/max', df['rater2'].min(), df['rater2'].max(), 'unique', len(df['rater2'].unique()))

print('yellowCards describe')
print(df['yellowCards'].describe())
print('yellowCards unique sample', sorted(df['yellowCards'].unique())[:10])

print('redCards describe')
print(df['redCards'].describe())
print('redCards unique sample', sorted(df['redCards'].unique())[:10])

