import pandas as pd
import numpy as np
import statsmodels.api as sm

DF = pd.read_csv('soccer.csv')

skin1 = DF['rater1']
skin2 = DF['nExp']
red_cards = DF['yellowCards']  # per info.json: red cards

games = DF['redCards']  # per info.json: number of games

analysis = pd.DataFrame({
    'skin1': skin1,
    'skin2': skin2,
    'red_cards': red_cards,
    'games': games,
}).dropna(subset=['skin1','skin2'])
analysis['skin_mean'] = (analysis['skin1'] + analysis['skin2'])/2
analysis['skin_group'] = np.where(analysis['skin_mean'] > 0.5, 'dark',
                          np.where(analysis['skin_mean'] < 0.5, 'light', 'mid'))
analysis['red_per_game'] = analysis['red_cards'] / analysis['games']

summary = analysis.groupby('skin_group').agg(
    n=('red_cards','size'),
    red_cards_sum=('red_cards','sum'),
    games_sum=('games','sum'),
)
summary['rate_overall'] = summary['red_cards_sum'] / summary['games_sum']
print(summary)

# Poisson GLM with offset
analysis = analysis[analysis['games']>0]
X = sm.add_constant(analysis['skin_mean'])
res = sm.GLM(analysis['red_cards'], X, family=sm.families.Poisson(), offset=np.log(analysis['games'])).fit(cov_type='HC3')

coef = res.params['skin_mean']
se = res.bse['skin_mean']
p = res.pvalues['skin_mean']
ci_low, ci_high = res.conf_int().loc['skin_mean']

rr = float(np.exp(coef))
rr_low = float(np.exp(ci_low))
rr_high = float(np.exp(ci_high))

# per 0.25 increment
rr_q = float(np.exp(coef*0.25))

print('\ncoef', coef, 'se', se, 'p', p)
print('RR per 1.0 increase', rr, 'CI', rr_low, rr_high)
print('RR per 0.25 increase', rr_q)

