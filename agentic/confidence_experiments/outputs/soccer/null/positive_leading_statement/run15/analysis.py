import pandas as pd
import numpy as np

path = 'soccer.csv'

df = pd.read_csv(path)

# compute average skin tone for players with photo ratings
skin = df[['rater1','rater2']].mean(axis=1)

df = df.assign(skin_mean=skin)

print('rows', len(df))
print('missing skin', df['skin_mean'].isna().mean())

# unique skin values
print('unique skin values', sorted(df['skin_mean'].dropna().unique())[:10], '... count', df['skin_mean'].nunique())

# distribution counts
print(df['skin_mean'].value_counts(dropna=False).sort_index())

# check redCards and games summary
print('redCards summary', df['redCards'].describe())
print('games summary', df['games'].describe())

# compute rate per game for each row
rate = df['redCards'] / df['games']
print('rate summary', rate.describe())

# categorize light vs dark
# using thresholds <=0.25 light, >=0.75 dark
cond_light = df['skin_mean'] <= 0.25
cond_dark = df['skin_mean'] >= 0.75

for label, cond in [('light', cond_light), ('dark', cond_dark)]:
    sub = df[cond]
    print(label, 'n', len(sub))
    print('redCards per game', (sub['redCards'].sum() / sub['games'].sum()))
    print('redCards per dyad mean', sub['redCards'].mean())

# also compare using threshold >0.5
cond_light2 = df['skin_mean'] < 0.5
cond_dark2 = df['skin_mean'] > 0.5
for label, cond in [('light2', cond_light2), ('dark2', cond_dark2)]:
    sub = df[cond]
    print(label, 'n', len(sub))
    print('redCards per game', (sub['redCards'].sum() / sub['games'].sum()))

