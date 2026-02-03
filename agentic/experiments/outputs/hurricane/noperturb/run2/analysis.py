import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('hurricane.csv')

# Basic preprocessing
_df['log_deaths'] = np.log1p(_df['alldeaths'])

# Models
models = {}
models['bivariate_masfem'] = smf.ols('log_deaths ~ masfem', data=_df).fit(cov_type='HC3')
models['controls_intensity'] = smf.ols(
    'log_deaths ~ masfem + wind + min + category + year',
    data=_df
).fit(cov_type='HC3')
models['controls_plus_damage'] = smf.ols(
    'log_deaths ~ masfem + wind + min + category + year + ndam15',
    data=_df
).fit(cov_type='HC3')
models['gender_binary_controls'] = smf.ols(
    'log_deaths ~ gender_mf + wind + min + category + year',
    data=_df
).fit(cov_type='HC3')

# Collect key results
summary_rows = []
for name, m in models.items():
    if 'masfem' in m.params.index:
        coef = m.params['masfem']
        pval = m.pvalues['masfem']
        summary_rows.append((name, 'masfem', coef, pval))
    if 'gender_mf' in m.params.index:
        coef = m.params['gender_mf']
        pval = m.pvalues['gender_mf']
        summary_rows.append((name, 'gender_mf', coef, pval))

summary = pd.DataFrame(summary_rows, columns=['model', 'term', 'coef', 'pvalue'])

print('Rows:', len(_df))
print('Missing values per column:')
print(_df.isna().sum())
print('\nCorrelation masfem vs deaths:', _df['masfem'].corr(_df['alldeaths']))
print('\nModel summaries (HC3 robust SEs):')
for name, m in models.items():
    print('\n===', name, '===')
    print(m.summary())

print('\nKey coefficients:')
print(summary.to_string(index=False))
