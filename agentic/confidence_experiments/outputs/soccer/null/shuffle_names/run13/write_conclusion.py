import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

path = 'soccer.csv'
df = pd.read_csv(path)

skin1 = 'rater1'
skin2 = 'nExp'
games = 'redCards'       # number of games in dyad (per metadata)
red_cards = 'yellowCards'  # number of red cards (per metadata)

use = df[[skin1, skin2, games, red_cards]].dropna().copy()
use['skin_avg'] = (use[skin1] + use[skin2]) / 2

# Define light/dark groups
use['skin_group'] = np.where(use['skin_avg'] <= 0.25, 'light',
                             np.where(use['skin_avg'] >= 0.75, 'dark', 'mid'))

summary = use.groupby('skin_group').agg(
    dyads=('skin_group', 'size'),
    total_red_cards=(red_cards, 'sum'),
    total_games=(games, 'sum')
)

rate_dark = summary.loc['dark', 'total_red_cards'] / summary.loc['dark', 'total_games']
rate_light = summary.loc['light', 'total_red_cards'] / summary.loc['light', 'total_games']
rate_ratio = rate_dark / rate_light

# Poisson regression with offset and robust SE
X = sm.add_constant(use['skin_avg'])
pois_model = sm.GLM(use[red_cards], X, family=sm.families.Poisson(), offset=np.log(use[games]))
pois_res = pois_model.fit(cov_type='HC0')
coef_p = pois_res.params['skin_avg']
pval_p = pois_res.pvalues['skin_avg']
irr_p = float(np.exp(coef_p))

# Negative binomial (default alpha) for sensitivity
nb_model = sm.GLM(use[red_cards], X, family=sm.families.NegativeBinomial(), offset=np.log(use[games]))
nb_res = nb_model.fit()
coef_nb = nb_res.params['skin_avg']
pval_nb = nb_res.pvalues['skin_avg']
irr_nb = float(np.exp(coef_nb))

response = 25  # Likert scale: low, indicating "No" given lack of evidence and slight negative association

explanation = (
    f"Using dyad-level data with available skin ratings (N={len(use):,} dyads), I averaged the two rater scores "
    f"to form a 0-1 skin tone measure and modeled red cards (column 'yellowCards') with a Poisson GLM using an "
    f"offset for the number of games (column 'redCards'). The skin-tone effect is negative and not statistically "
    f"significant (IRR about {irr_p:.3f}, p about {pval_p:.3f}; robust SE). A negative-binomial sensitivity check gives a similar "
    f"result (IRR about {irr_nb:.3f}, p about {pval_nb:.3f}). In a simple rate comparison, dark-tone dyads (skin_avg >= 0.75) "
    f"have a lower red-card rate (about {rate_dark:.5f} per game) than light-tone dyads (skin_avg <= 0.25; about {rate_light:.5f} per game), "
    f"rate ratio about {rate_ratio:.3f}. Overall, there is no evidence that darker-skinned players are more likely to receive red cards; "
    f"if anything, the point estimates are slightly lower for darker skin tones."
)

with open('conclusion.txt', 'w') as f:
    json.dump({"response": response, "explanation": explanation}, f)

print('Wrote conclusion.txt')
