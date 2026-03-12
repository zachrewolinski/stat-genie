import pandas as pd
import numpy as np
import statsmodels.api as sm


df = pd.read_csv('soccer.csv')

# identify games column (redCards) and skin ratings
# mean skin rating from two raters
skin = df[['rater1','nExp']].mean(axis=1)

df = df.assign(mean_skin=skin)

# total red-card events (straight red + second yellow)
# The two rare columns are meanExp and yellowCards; sum to capture all red-card events
red_total = df['meanExp'] + df['yellowCards']

df = df.assign(red_total=red_total)

# games exposure (column named redCards equals sum of three outcome columns)
# ensure games > 0

df = df[df['redCards'] > 0].copy()

# drop rows with missing skin
_df = df.dropna(subset=['mean_skin']).copy()

print('rows with skin', _df.shape[0])

# define skin categories
bins = [-np.inf, 0.25, 0.5, np.inf]
labels = ['light', 'medium', 'dark']
_df['skin_cat'] = pd.cut(_df['mean_skin'], bins=bins, labels=labels)

# dyad-level rates by skin cat
rate_by_cat = _df.groupby('skin_cat').apply(lambda g: pd.Series({
    'red_total': g['red_total'].sum(),
    'games': g['redCards'].sum(),
    'rate_per_game': g['red_total'].sum() / g['redCards'].sum()
}))
print('\nDyad-level rate by skin category:')
print(rate_by_cat)

# player-level aggregation
player_df = _df.groupby('photoID').agg(
    mean_skin=('mean_skin','first'),
    red_total=('red_total','sum'),
    games=('redCards','sum')
).reset_index()
player_df['skin_cat'] = pd.cut(player_df['mean_skin'], bins=bins, labels=labels)

player_rate = player_df.groupby('skin_cat').apply(lambda g: pd.Series({
    'players': g.shape[0],
    'red_total': g['red_total'].sum(),
    'games': g['games'].sum(),
    'rate_per_game': g['red_total'].sum() / g['games'].sum()
}))
print('\nPlayer-level rate by skin category:')
print(player_rate)

# Poisson regression at dyad level with exposure
X = sm.add_constant(_df['mean_skin'])
model = sm.GLM(_df['red_total'], X, family=sm.families.Poisson(), offset=np.log(_df['redCards']))
res = model.fit()
print('\nDyad-level Poisson with offset:')
print(res.summary())

# Poisson regression at player level
X_p = sm.add_constant(player_df['mean_skin'])
model_p = sm.GLM(player_df['red_total'], X_p, family=sm.families.Poisson(), offset=np.log(player_df['games']))
res_p = model_p.fit()
print('\nPlayer-level Poisson with offset:')
print(res_p.summary())

# compute rate ratio for dark vs light using player-level data
light_rate = player_rate.loc['light','rate_per_game']
dark_rate = player_rate.loc['dark','rate_per_game']
print('\nRate ratio dark/light (player-level):', dark_rate/light_rate if light_rate>0 else np.nan)

