import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Create proportion-like outcome; avoid division by zero (sockets min 2)
_df = _df.copy()
_df['amtl_rate'] = _df['num_amtl'] / _df['sockets']
_df['human'] = (_df['genus'] == 'Homo sapiens').astype(int)

# Ensure categorical types
_df['tooth_class'] = _df['tooth_class'].astype('category')
_df['genus'] = _df['genus'].astype('category')

# Weighted least squares with cluster-robust SE by specimen
# Model 1: human vs non-human, controlling for age, sex, tooth class
formula1 = 'amtl_rate ~ human + age + prob_male + C(tooth_class)'
model1 = smf.wls(formula1, data=_df, weights=_df['sockets']).fit(cov_type='cluster', cov_kwds={'groups': _df['specimen']})

# Model 2: genus categorical with Homo sapiens as reference
# Relevel by setting category order
if 'Homo sapiens' in _df['genus'].cat.categories:
    _df['genus'] = _df['genus'].cat.reorder_categories(['Homo sapiens'] + [g for g in _df['genus'].cat.categories if g != 'Homo sapiens'], ordered=False)

formula2 = 'amtl_rate ~ C(genus) + age + prob_male + C(tooth_class)'
model2 = smf.wls(formula2, data=_df, weights=_df['sockets']).fit(cov_type='cluster', cov_kwds={'groups': _df['specimen']})

# Summaries
print('N rows', len(_df), 'N specimens', _df['specimen'].nunique())
print('\nModel 1 (human vs non-human)')
print(model1.summary().tables[1])

print('\nModel 2 (genus categorical; reference=Homo sapiens)')
print(model2.summary().tables[1])

# Compute mean amtl_rate by genus for context
print('\nMean amtl_rate by genus')
print(_df.groupby('genus')['amtl_rate'].mean())

# Provide a simple effect size: difference between human and non-human mean (weighted by sockets)
weighted_mean = _df.groupby('human').apply(lambda d: np.average(d['amtl_rate'], weights=d['sockets']))
print('\nWeighted mean amtl_rate by human (0=non-human,1=human)')
print(weighted_mean)
