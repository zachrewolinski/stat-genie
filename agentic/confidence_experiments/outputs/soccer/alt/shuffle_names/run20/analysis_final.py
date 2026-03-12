import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


df = pd.read_csv('soccer.csv')

# Skin tone average from two raters
skin = df['rater1'].add(df['nExp']).div(2)

games = df['redCards']

candidates = ['meanExp', 'yellowCards']

results = {}
for col in candidates:
    data = pd.DataFrame({
        'skin': skin,
        'games': games,
        'y': df[col]
    }).dropna()

    # per-game rate
    data['rate'] = data['y'] / data['games']

    dark = data['skin'] > 0.5
    light = data['skin'] < 0.5

    # t-test for rate difference
    tstat, pval = stats.ttest_ind(data.loc[dark, 'rate'], data.loc[light, 'rate'], equal_var=False)

    # Poisson regression with offset (rate modeling)
    X = sm.add_constant(data['skin'])
    model = sm.GLM(data['y'], X, family=sm.families.Poisson(), offset=np.log(data['games']))
    res = model.fit()

    results[col] = {
        'dark_n': int(dark.sum()),
        'light_n': int(light.sum()),
        'dark_rate': data.loc[dark, 'rate'].mean(),
        'light_rate': data.loc[light, 'rate'].mean(),
        'rate_diff': data.loc[dark, 'rate'].mean() - data.loc[light, 'rate'].mean(),
        'ttest_p': pval,
        'poisson_coef_skin': res.params['skin'],
        'poisson_p_skin': res.pvalues['skin'],
    }

print(results)
