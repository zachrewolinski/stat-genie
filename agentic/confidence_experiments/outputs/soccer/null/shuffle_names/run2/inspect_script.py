import pandas as pd

path = 'soccer.csv'
df = pd.read_csv(path)
print(df.head())
print('columns', df.columns.tolist())
print('rater1 unique', sorted(df['rater1'].dropna().unique())[:10])
print('nExp unique', sorted(df['nExp'].dropna().unique())[:10])
print('yellowCards summary', df['yellowCards'].describe())
print('redCards summary', df['redCards'].describe())
print('meanExp unique', sorted(df['meanExp'].dropna().unique())[:10])
print('yellowReds summary', df['yellowReds'].describe())
print('games? redCards min max', df['redCards'].min(), df['redCards'].max())
