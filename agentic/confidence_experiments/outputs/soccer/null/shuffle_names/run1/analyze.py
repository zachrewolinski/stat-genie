import pandas as pd
import numpy as np
import statsmodels.api as sm


df = pd.read_csv('soccer.csv')

# infer key variables
# red card counts are in yellowCards (0-2), games in redCards (1-47)
red_cards = df['yellowCards']
games = df['redCards']

# skin tone: average of two raters (0=very light, 1=very dark)
skin = df[['rater1', 'nExp']].mean(axis=1)

# use player identifier to aggregate to player level
player_id = df['photoID']

player_df = (
    df.assign(red_cards=red_cards, games=games, skin=skin)
      .dropna(subset=['skin'])
      .groupby(player_id, as_index=False)
      .agg(red_cards=('red_cards','sum'), games=('games','sum'), skin=('skin','mean'))
)

# rate per game
player_df['red_rate'] = player_df['red_cards'] / player_df['games']

# describe counts by skin categories (light <=0.25, dark >=0.75)
light = player_df[player_df['skin'] <= 0.25]
dark = player_df[player_df['skin'] >= 0.75]

summary = {
    'n_players': len(player_df),
    'light_n': len(light),
    'dark_n': len(dark),
    'light_rate_mean': light['red_rate'].mean(),
    'dark_rate_mean': dark['red_rate'].mean(),
}

# Poisson regression with offset for games, robust SE
X = sm.add_constant(player_df['skin'])
model = sm.GLM(player_df['red_cards'], X, family=sm.families.Poisson(), offset=np.log(player_df['games']))
res = model.fit(cov_type='HC3')

print('summary', summary)
print('coef', res.params)
print('pvalues', res.pvalues)
print('confint', res.conf_int())

