import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('soccer.csv')
print('rows', len(df))
print(df[['rater1','rater2','redCards','games']].head())

# compute skin tone average
df['skin'] = df[['rater1','rater2']].mean(axis=1)
print('skin missing', df['skin'].isna().mean())
print(df['skin'].describe())

# define dark/light categories: use top/bottom tertile maybe? But let's inspect distribution
print(df['skin'].value_counts(dropna=False).head())

# compute red cards per game
df['red_per_game'] = df['redCards'] / df['games']

# overall correlation
print('corr skin-red_per_game', df[['skin','red_per_game']].corr().iloc[0,1])

# simple group: light <=0.25 (very light/light), dark >=0.75 (dark/very dark) using 5-point scale mapping? but values normalized to 0-1; maybe categories 0,0.25,0.5,0.75,1
print('unique skin values', sorted(df['skin'].dropna().unique())[:10])

# map to discrete categories if possible
# compute categories based on average rounding to nearest 0.25
df['skin_round'] = df['skin'].round(2)
print('skin_round unique count', df['skin_round'].nunique())

# create light vs dark using thresholds
light = df['skin'] <= 0.25
dark = df['skin'] >= 0.75

for label, mask in [('light', light), ('dark', dark)]:
    subset = df[mask]
    print(label, 'n', len(subset))
    print('mean redCards', subset['redCards'].mean(), 'mean red_per_game', subset['red_per_game'].mean())
    print('any red card rate', (subset['redCards']>0).mean())

# middle group
mid = (df['skin'] > 0.25) & (df['skin'] < 0.75)
print('mid n', mid.sum())

# maybe use quantiles (median split)
median = df['skin'].median()
print('median', median)

low = df['skin'] <= median
high = df['skin'] > median
for label, mask in [('low', low), ('high', high)]:
    subset = df[mask]
    print(label, 'n', len(subset))
    print('mean redCards', subset['redCards'].mean(), 'mean red_per_game', subset['red_per_game'].mean())
    print('any red card rate', (subset['redCards']>0).mean())
