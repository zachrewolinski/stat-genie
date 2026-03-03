import pandas as pd
import statsmodels.formula.api as smf

_df = pd.read_csv('amtl.csv')
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

formula = 'amtl_count ~ C(genus_group, Treatment(reference="Homo sapiens")) + sockets_n + age_at_death + sex_prob + C(tooth_class)'
model = smf.ols(formula, data=_df)
res = model.fit(cov_type='cluster', cov_kwds={'groups': _df['specimen_id']})
print(res.summary())
print('\nGenus group differences vs Homo:')
for name in res.params.filter(like='C(genus_group').index:
    print(name, res.params[name], res.pvalues[name])
