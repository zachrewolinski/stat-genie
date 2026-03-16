import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats


df = pd.read_csv('soccer.csv')
red_cards = df['yellowCards']
games = df['redCards']
skin = df[['rater1','nExp']].mean(axis=1)
player_id = df['photoID']

player_df = (
    df.assign(red_cards=red_cards, games=games, skin=skin)
      .dropna(subset=['skin'])
      .groupby(player_id, as_index=False)
      .agg(red_cards=('red_cards','sum'), games=('games','sum'), skin=('skin','mean'))
)

player_df['red_rate'] = player_df['red_cards'] / player_df['games']

# Poisson regression with offset log(games)
X = sm.add_constant(player_df['skin'])
model = sm.GLM(player_df['red_cards'], X, family=sm.families.Poisson(), offset=np.log(player_df['games']))
res = model.fit(cov_type='HC3')

# median split
median_skin = player_df['skin'].median()
light = player_df[player_df['skin'] <= median_skin]
dark = player_df[player_df['skin'] > median_skin]

rate_diff = dark['red_rate'].mean() - light['red_rate'].mean()
rate_ratio = dark['red_rate'].mean() / light['red_rate'].mean() if light['red_rate'].mean() > 0 else np.nan

# t-test on rates (Welch)
stat, pval = stats.ttest_ind(dark['red_rate'], light['red_rate'], equal_var=False)

print('n_players', len(player_df))
print('median_skin', median_skin)
print('light_n', len(light), 'dark_n', len(dark))
print('light_rate_mean', light['red_rate'].mean())
print('dark_rate_mean', dark['red_rate'].mean())
print('rate_diff', rate_diff)
print('rate_ratio', rate_ratio)
print('t_pval', pval)
print('poisson_coef', res.params['skin'])
print('poisson_pval', res.pvalues['skin'])
print('poisson_confint', res.conf_int().loc['skin'].tolist())

