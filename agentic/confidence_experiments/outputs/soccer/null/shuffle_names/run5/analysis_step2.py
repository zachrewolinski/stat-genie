import pandas as pd
import numpy as np
import statsmodels.api as sm

_df = pd.read_csv('soccer.csv')

# Identify skin ratings
skin1 = 'rater1'
skin2 = 'nExp'
_df['skin_mean'] = _df[[skin1, skin2]].mean(axis=1)

# Identify counts
# games column inferred
_df['games'] = _df['redCards']
# direct red cards? assume yellowCards
_df['red_direct'] = _df['yellowCards']
# second-yellow red cards? assume meanExp
_df['red_second'] = _df['meanExp']
_df['red_total'] = _df['red_direct'] + _df['red_second']

# aggregate to player level
player_col = 'photoID'
agg = _df.groupby(player_col).agg(
    games=('games','sum'),
    red_direct=('red_direct','sum'),
    red_second=('red_second','sum'),
    red_total=('red_total','sum'),
    skin_mean=('skin_mean','mean'),
    skin1=('rater1','mean'),
    skin2=('nExp','mean'),
)

print('player count', len(agg))
print(agg[['games','red_total','skin_mean']].describe())

# distribution of skin_mean
print('skin_mean value counts')
print(agg['skin_mean'].round(3).value_counts().sort_index().head(10))
print(agg['skin_mean'].round(3).value_counts().sort_index().tail(10))

# define light/dark (exclude mid): light<=0.25, dark>=0.75
light = agg[agg['skin_mean'] <= 0.25]
dark = agg[agg['skin_mean'] >= 0.75]
print('light n', len(light), 'dark n', len(dark))
print('light red_total per game', (light['red_total'].sum()/light['games'].sum()))
print('dark red_total per game', (dark['red_total'].sum()/dark['games'].sum()))

# Poisson regression at player level with offset log(games)
# Use red_total as outcome
agg = agg[agg['games'] > 0].copy()
agg['log_games'] = np.log(agg['games'])
X = sm.add_constant(agg['skin_mean'])
model = sm.GLM(agg['red_total'], X, family=sm.families.Poisson(), offset=agg['log_games'])
res = model.fit()
print(res.summary())

# Also do negative binomial if overdispersion
from statsmodels.discrete.discrete_model import NegativeBinomial

nb_model = NegativeBinomial(agg['red_total'], X, offset=agg['log_games'])
nb_res = nb_model.fit(disp=0)
print(nb_res.summary())

# simple correlation of skin_mean with red rate
agg['red_rate'] = agg['red_total'] / agg['games']
print('corr', agg['skin_mean'].corr(agg['red_rate']))

