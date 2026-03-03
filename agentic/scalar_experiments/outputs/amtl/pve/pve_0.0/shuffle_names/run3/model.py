import pandas as pd
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Rename for clarity
_df = _df.rename(columns={
    'sockets':'tooth_class',
    'prob_male':'specimen_id',
    'genus':'amtl_value',
    'age':'observable_sockets',
    'pop':'age_est',
    'num_amtl':'age_uncert',
    'stdev_age':'prob_male',
    'tooth_class':'genus',
    'specimen':'region'
})

_df['human'] = (_df['genus'] == 'Homo sapiens').astype(int)

# Fit OLS with cluster-robust SE by specimen_id
model = smf.ols('amtl_value ~ human + age_est + prob_male + C(tooth_class)', data=_df)
res = model.fit(cov_type='cluster', cov_kwds={'groups': _df['specimen_id']})

print(res.summary())

# Extract human coefficient
coef = res.params['human']
pval = res.pvalues['human']
print('\nHuman coef:', coef)
print('Human p-value:', pval)
