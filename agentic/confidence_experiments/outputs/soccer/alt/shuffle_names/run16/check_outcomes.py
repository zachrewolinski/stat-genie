import pandas as pd
import numpy as np
import statsmodels.api as sm


df = pd.read_csv('soccer.csv')

# mean skin
skin = df[['rater1','nExp']].mean(axis=1)
df = df.assign(mean_skin=skin)

# use games exposure
_df = df[df['redCards'] > 0].dropna(subset=['mean_skin']).copy()

outcomes = ['meanExp','yellowCards','meanExp_plus_yellowCards']

_df['meanExp_plus_yellowCards'] = _df['meanExp'] + _df['yellowCards']

for outcome in outcomes:
    X = sm.add_constant(_df['mean_skin'])
    model = sm.GLM(_df[outcome], X, family=sm.families.Poisson(), offset=np.log(_df['redCards']))
    res = model.fit()
    coef = res.params['mean_skin']
    pval = res.pvalues['mean_skin']
    print(outcome, 'coef', coef, 'p', pval)
