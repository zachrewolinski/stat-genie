import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest

# Load data

df = pd.read_csv('soccer.csv')

# Identify columns
# Skin tone ratings
r1 = 'rater1'
r2 = 'nExp'
# Red cards count (per description)
red_cards_col = 'yellowCards'
# Games in dyad (exposure)
games_col = 'redCards'

# Prepare dataset
sub = df[[r1, r2, red_cards_col, games_col]].copy()
sub = sub.dropna(subset=[r1, r2, red_cards_col, games_col])
sub = sub[sub[games_col] > 0]

# Average skin tone
sub['skin_tone'] = sub[[r1, r2]].mean(axis=1)

# Red card rate per game
sub['red_rate'] = sub[red_cards_col] / sub[games_col]

print('n', len(sub))
print(sub[['skin_tone', red_cards_col, games_col, 'red_rate']].describe())

# Poisson regression with offset log(games)
X = sm.add_constant(sub['skin_tone'])
model = sm.GLM(sub[red_cards_col], X, family=sm.families.Poisson(), offset=np.log(sub[games_col]))
res = model.fit(cov_type='HC3')
print(res.summary())

# Effect size: rate ratio for skin tone from 0 to 1
beta = res.params['skin_tone']
rr = np.exp(beta)
print('Rate ratio (skin tone 0->1):', rr)

# Also compare dark vs light groups: quartiles
q1 = sub['skin_tone'].quantile(0.25)
q3 = sub['skin_tone'].quantile(0.75)
light = sub[sub['skin_tone'] <= q1]
dark = sub[sub['skin_tone'] >= q3]
light_rate = light[red_cards_col].sum() / light[games_col].sum()
dark_rate = dark[red_cards_col].sum() / dark[games_col].sum()
print('light_rate', light_rate, 'dark_rate', dark_rate)

# z-test for proportions (approx) using total red cards and games
count = np.array([dark[red_cards_col].sum(), light[red_cards_col].sum()])
obs = np.array([dark[games_col].sum(), light[games_col].sum()])
stat, pval = proportions_ztest(count, obs)
print('z-test', stat, pval)
