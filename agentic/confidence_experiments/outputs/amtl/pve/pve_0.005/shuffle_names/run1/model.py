import pandas as pd
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Map columns based on info.json descriptions
_df = _df.rename(columns={
    'sockets': 'tooth_class',
    'prob_male': 'specimen_id',
    'genus': 'amtl_count',
    'age': 'sockets_n',
    'pop': 'age_at_death',
    'num_amtl': 'age_uncert',
    'stdev_age': 'sex_prob',
    'tooth_class': 'genus_group',
    'specimen': 'region',
})

# Outcome: AMTL rate
_df['amtl_rate'] = _df['amtl_count'] / _df['sockets_n']

# Build model: weighted least squares with cluster-robust SE by specimen
formula = 'amtl_rate ~ C(genus_group, Treatment(reference="Homo sapiens")) + age_at_death + sex_prob + C(tooth_class)'
model = smf.wls(formula, data=_df, weights=_df['sockets_n'])
res = model.fit(cov_type='cluster', cov_kwds={'groups': _df['specimen_id']})

print(res.summary())

# Extract coefficients for non-human genera vs Homo
coef = res.params.filter(like='C(genus_group')
se = res.bse[coef.index]
print('\nGenus group differences vs Homo (negative => lower than Homo):')
for name in coef.index:
    print(name, 'coef', coef[name], 'se', se[name], 'p', res.pvalues[name])

# Compute adjusted mean rates by genus_group at mean covariates and tooth_class distribution
# Use average marginal prediction over observed data but switching genus_group
means = {}
for g in _df['genus_group'].unique():
    tmp = _df.copy()
    tmp['genus_group'] = g
    # predicted mean
    pred = res.predict(tmp)
    means[g] = pred.mean()

print('\nAdjusted mean amtl_rate by genus_group (avg over covariates):')
for k,v in means.items():
    print(k, v)

