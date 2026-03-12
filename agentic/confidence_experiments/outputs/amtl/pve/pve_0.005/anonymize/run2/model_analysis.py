import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

_df = pd.read_csv('amtl.csv')
_df['ratio'] = _df['feature3'] / _df['feature4']
_df['is_human'] = (_df['feature8'] == 'Homo sapiens').astype(int)

# Model 1: ratio as outcome
model_ratio = smf.ols('ratio ~ is_human + feature5 + feature7 + C(feature1)', data=_df).fit(cov_type='HC3')

# Model 2: counts as outcome, control for observable sockets
model_count = smf.ols('feature3 ~ is_human + feature5 + feature7 + C(feature1) + feature4', data=_df).fit(cov_type='HC3')

print('Model ratio (robust HC3)')
print(model_ratio.summary().tables[1])
print('\nModel count (robust HC3)')
print(model_count.summary().tables[1])

# Extract key stats
coef_ratio = model_ratio.params['is_human']
p_ratio = model_ratio.pvalues['is_human']
coef_count = model_count.params['is_human']
p_count = model_count.pvalues['is_human']

print('\nKey stats')
print('ratio coef', coef_ratio, 'p', p_ratio)
print('count coef', coef_count, 'p', p_count)

# Means (unadjusted) for context
print('\nUnadjusted means')
print(_df.groupby('feature8')['ratio'].mean())
print(_df.groupby('feature8')['feature3'].mean())

# Additional: human vs non-human with all genus categories
model_genus = smf.ols('ratio ~ C(feature8) + feature5 + feature7 + C(feature1)', data=_df).fit(cov_type='HC3')
print('\nModel with genus categorical (ratio)')
print(model_genus.summary().tables[1])

