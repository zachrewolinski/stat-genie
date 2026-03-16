import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

DF = pd.read_csv('amtl.csv')
DF = DF.rename(columns={
    'genus': 'amtl_count',
    'age': 'sockets_count',
    'pop': 'age_at_death',
    'stdev_age': 'sex_prob_male',
    'tooth_class': 'genus_cat',
    'sockets': 'tooth_class',
    'prob_male': 'specimen_id'
})
DF['amtl_rate'] = DF['amtl_count'] / DF['sockets_count']

# Mean rates by genus
means = DF.groupby('genus_cat')['amtl_rate'].mean().sort_index()
print('mean rates by genus:\n', means)

# Weighted least squares model
model = smf.wls(
    'amtl_rate ~ C(genus_cat) + age_at_death + sex_prob_male + C(tooth_class)',
    data=DF,
    weights=DF['sockets_count']
).fit(cov_type='HC3')

print('WLS coefficients for genus categories:')
for term in ['C(genus_cat)[T.Pan]','C(genus_cat)[T.Papio]','C(genus_cat)[T.Pongo]']:
    print(term, model.params[term], model.pvalues[term])

