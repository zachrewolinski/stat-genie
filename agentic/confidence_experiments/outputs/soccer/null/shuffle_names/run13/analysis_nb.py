import pandas as pd
import numpy as np
import statsmodels.api as sm

path = 'soccer.csv'
df = pd.read_csv(path)

skin1 = 'rater1'
skin2 = 'nExp'
games = 'redCards'
red_cards = 'yellowCards'

use = df[[skin1, skin2, games, red_cards]].dropna().copy()
use['skin_avg'] = (use[skin1] + use[skin2]) / 2

X = sm.add_constant(use['skin_avg'])

# Negative Binomial GLM with offset
nb_model = sm.GLM(use[red_cards], X, family=sm.families.NegativeBinomial(), offset=np.log(use[games]))
nb_res = nb_model.fit()
coef = nb_res.params['skin_avg']
pval = nb_res.pvalues['skin_avg']
irr = np.exp(coef)
print('NB coef', coef, 'IRR', irr, 'p-value', pval)

# Poisson with robust SE
pois_model = sm.GLM(use[red_cards], X, family=sm.families.Poisson(), offset=np.log(use[games]))
pois_res = pois_model.fit(cov_type='HC0')
coef_p = pois_res.params['skin_avg']
pval_p = pois_res.pvalues['skin_avg']
irr_p = np.exp(coef_p)
print('Poisson robust coef', coef_p, 'IRR', irr_p, 'p-value', pval_p)

