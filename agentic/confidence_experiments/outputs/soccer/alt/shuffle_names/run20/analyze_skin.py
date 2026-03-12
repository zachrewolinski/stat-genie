import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


df = pd.read_csv('soccer.csv')
# identify skin tone columns
skin1 = df['rater1']
skin2 = df['nExp']
skin = skin1.add(skin2).div(2)

print('skin avg unique', sorted(skin.unique())[:10], 'nunique', skin.nunique())
print('skin avg value counts')
print(skin.value_counts().sort_index())

# choose red card column candidates
candidates = ['meanExp','yellowCards']

# use games column
games = df['redCards']

for col in candidates:
    # rate per game
    rate = df[col] / games
    # group by skin category
    dark = skin > 0.5
    light = skin < 0.5
    # exclude middle 0.5
    print('\nCandidate', col)
    print('dark count', dark.sum(), 'light count', light.sum())
    print('dark mean rate', rate[dark].mean(), 'light mean rate', rate[light].mean())
    # t-test
    tstat, pval = stats.ttest_ind(rate[dark], rate[light], equal_var=False)
    print('t-test p', pval)

    # Poisson regression with offset for games
    X = sm.add_constant(skin)
    model = sm.GLM(df[col], X, family=sm.families.Poisson(), offset=np.log(games))
    res = model.fit()
    print('Poisson coef skin', res.params[1], 'p', res.pvalues[1])
