import pandas as pd
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('amtl.csv')

# Map columns to semantic names
df = df.rename(
    columns={
        'genus': 'amtl',          # AMTL measure (continuous)
        'tooth_class': 'genus',   # genus categories
        'sockets': 'tooth_class', # tooth class categories
        'pop': 'age_at_death',    # age at death
        'stdev_age': 'prob_male', # sex estimate (probability male)
        'age': 'sockets_count',   # number of observable sockets
        'num_amtl': 'age_uncertainty',
        'prob_male': 'specimen_id',
        'specimen': 'population',
    }
)

# Ensure categories
df['genus'] = df['genus'].astype('category')
df['tooth_class'] = df['tooth_class'].astype('category')

# Use Homo sapiens as baseline for genus
if 'Homo sapiens' in df['genus'].cat.categories:
    df['genus'] = df['genus'].cat.reorder_categories(
        ['Homo sapiens']
        + [g for g in df['genus'].cat.categories if g != 'Homo sapiens'],
        ordered=False,
    )

# Model: AMTL ~ genus + age + sex + tooth_class
model = smf.ols(
    'amtl ~ C(genus) + age_at_death + prob_male + C(tooth_class)', data=df
).fit(cov_type='HC3')

print(model.summary())

# Extract genus coefficients (relative to Homo sapiens)
params = model.params
pvalues = model.pvalues

print('\nGenus effects vs Homo sapiens (HC3 robust):')
for term in params.index:
    if term.startswith('C(genus)'):
        print(term, 'coef', params[term], 'p', pvalues[term])

# Also run model with sockets_count as additional covariate
model2 = smf.ols(
    'amtl ~ C(genus) + age_at_death + prob_male + C(tooth_class) + sockets_count',
    data=df,
).fit(cov_type='HC3')

print('\nModel with sockets_count covariate:')
print(model2.summary())

print('\nGenus effects vs Homo sapiens (HC3 robust) with sockets_count:')
for term in model2.params.index:
    if term.startswith('C(genus)'):
        print(term, 'coef', model2.params[term], 'p', model2.pvalues[term])
