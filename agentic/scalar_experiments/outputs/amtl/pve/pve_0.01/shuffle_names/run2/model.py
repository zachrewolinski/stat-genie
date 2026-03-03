import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
DF = pd.read_csv('amtl.csv')

# Rename columns for clarity
DF = DF.rename(columns={
    'genus': 'amtl_count',
    'age': 'sockets_count',
    'pop': 'age_at_death',
    'stdev_age': 'sex_prob_male',
    'tooth_class': 'genus_cat',
    'sockets': 'tooth_class',
    'prob_male': 'specimen_id'
})

# Compute AMTL rate
DF['amtl_rate'] = DF['amtl_count'] / DF['sockets_count']

# Weighted least squares for rate
model = smf.wls(
    'amtl_rate ~ C(genus_cat) + age_at_death + sex_prob_male + C(tooth_class)',
    data=DF,
    weights=DF['sockets_count']
).fit(cov_type='HC3')

print(model.summary())

# Compute Homo sapiens vs each non-human by releveling
results = {}
for base in DF['genus_cat'].unique():
    tmp = DF.copy()
    # set categorical order for baseline
    categories = [base] + [c for c in DF['genus_cat'].unique() if c != base]
    tmp['genus_cat'] = pd.Categorical(tmp['genus_cat'], categories=categories, ordered=True)
    m = smf.wls(
        'amtl_rate ~ C(genus_cat) + age_at_death + sex_prob_male + C(tooth_class)',
        data=tmp,
        weights=tmp['sockets_count']
    ).fit(cov_type='HC3')
    if base != 'Homo sapiens':
        coef = m.params.get('C(genus_cat)[T.Homo sapiens]')
        pval = m.pvalues.get('C(genus_cat)[T.Homo sapiens]')
        results[base] = (coef, pval)

print('Homo sapiens vs baseline results:', results)

# Alternative model: amtl_count as outcome with sockets_count as covariate
model2 = smf.ols(
    'amtl_count ~ C(genus_cat) + sockets_count + age_at_death + sex_prob_male + C(tooth_class)',
    data=DF
).fit(cov_type='HC3')

print(model2.summary())
print('model2 genus params', model2.params.filter(like='C(genus_cat)').to_string())

