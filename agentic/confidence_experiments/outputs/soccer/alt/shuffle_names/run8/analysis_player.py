import pandas as pd
import numpy as np
import statsmodels.api as sm

path = 'soccer.csv'
df = pd.read_csv(path)

red_card_col = 'yellowCards'
games_col = 'redCards'
skin_cols = ['rater1', 'nExp']
player_id = 'photoID'

sub = df[[player_id, red_card_col, games_col] + skin_cols].dropna()
sub['skin_tone'] = sub[skin_cols].mean(axis=1)
sub = sub[(sub[games_col] > 0) & (sub[red_card_col] >= 0)]

# Aggregate per player
agg = sub.groupby(player_id).agg(
    red_cards=(red_card_col, 'sum'),
    games=(games_col, 'sum'),
    skin_tone=('skin_tone', 'mean'),
)
agg = agg[agg['games'] > 0]

# Poisson regression at player level
X = sm.add_constant(agg['skin_tone'])
model = sm.GLM(agg['red_cards'], X, family=sm.families.Poisson(), offset=np.log(agg['games']))
res = model.fit(cov_type='HC0')

coef = res.params['skin_tone']
rate_ratio = float(np.exp(coef))
print('Players:', len(agg))
print('coef:', coef, 'p:', res.pvalues['skin_tone'], 'RR:', rate_ratio)
