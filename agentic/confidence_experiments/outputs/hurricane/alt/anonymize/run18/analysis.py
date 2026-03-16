import pandas as pd
import numpy as np
import statsmodels.api as sm

_df = pd.read_csv('hurricane.csv')
_df['log_deaths'] = np.log1p(_df['feature8'])
_df['masfem'] = _df['feature4']
_df['masfem_mt'] = _df['feature12']
_df['female_name'] = _df['feature6']

controls = ['feature7', 'feature13', 'feature5', 'feature2']


def fit_ols(y, x_cols):
    X = _df[x_cols].copy()
    X = sm.add_constant(X, has_constant='add')
    return sm.OLS(_df[y], X).fit(cov_type='HC3')

results = []

for label, var in [('masfem', 'masfem'), ('masfem_mt', 'masfem_mt'), ('female_name', 'female_name')]:
    m1 = fit_ols('log_deaths', [var])
    m2 = fit_ols('log_deaths', [var] + controls)
    for name, m in [(f'{label}_only', m1), (f'{label}_controls', m2)]:
        coef = m.params[var]
        pval = m.pvalues[var]
        se = m.bse[var]
        results.append((name, coef, se, pval, m.rsquared))

corrs = {
    'masfem': _df[['masfem', 'log_deaths']].corr().iloc[0, 1],
    'masfem_mt': _df[['masfem_mt', 'log_deaths']].corr().iloc[0, 1],
    'female_name': _df[['female_name', 'log_deaths']].corr().iloc[0, 1],
}

print('N', len(_df))
print('correlations', corrs)
for r in results:
    print(r)
