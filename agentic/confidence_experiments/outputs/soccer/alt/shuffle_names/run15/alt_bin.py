import pandas as pd
import numpy as np
import statsmodels.api as sm

DF = pd.read_csv('soccer.csv')

skin1 = DF['rater1']
skin2 = DF['nExp']
red_cards = DF['yellowCards']
games = DF['redCards']

analysis = pd.DataFrame({
    'skin1': skin1,
    'skin2': skin2,
    'red_cards': red_cards,
    'games': games,
}).dropna(subset=['skin1','skin2'])
analysis['skin_mean'] = (analysis['skin1'] + analysis['skin2'])/2
analysis['skin_group'] = np.where(analysis['skin_mean'] > 0.5, 'dark',
                          np.where(analysis['skin_mean'] < 0.5, 'light', 'mid'))
analysis = analysis[analysis['skin_group'].isin(['light','dark'])]
analysis['dark'] = (analysis['skin_group']=='dark').astype(int)

X = sm.add_constant(analysis['dark'])
model = sm.GLM(analysis['red_cards'], X, family=sm.families.Poisson(),
               offset=np.log(analysis['games']))
res = model.fit(cov_type='HC3')
print(res.summary())
