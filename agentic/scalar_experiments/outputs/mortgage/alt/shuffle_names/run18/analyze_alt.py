import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportions_ztest

_df = pd.read_csv('mortgage.csv')

# If deny is outcome
if 'deny' in _df.columns and 'female' in _df.columns:
    rate = _df.groupby('female')['deny'].mean()
    print('Denial rates by female:', rate)
    succ = _df.groupby('female')['deny'].sum().reindex([0,1]).fillna(0)
    n = _df['female'].value_counts().reindex([0,1]).fillna(0)
    stat, pval = proportions_ztest(succ, n)
    print('Two-proportion z-test on deny stat', stat, 'p', pval)

    # Logistic regression for deny
    controls = [c for c in _df.columns if c not in {'deny','female','accept'} and _df[c].dtype != 'O']
    if 'Unnamed: 0' in controls:
        controls.remove('Unnamed: 0')
    formula = 'deny ~ female'
    if controls:
        formula += ' + ' + ' + '.join(controls)
    model = smf.logit(formula=formula, data=_df).fit(disp=False)
    print(model.summary())
    print('female coef', model.params['female'], 'p', model.pvalues['female'], 'odds ratio', np.exp(model.params['female']))
