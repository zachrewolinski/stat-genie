import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('amtl.csv')

# Fit OLS with categorical genus and tooth class
model = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['specimen']}
)

print(model.summary())

# Compute adjusted predictions by genus (g-computation):
# For each genus, set genus to that value for all rows, keep other covariates as observed,
# compute mean predicted num_amtl.

genera = df['genus'].unique()

adj_means = {}
for g in genera:
    temp = df.copy()
    temp['genus'] = g
    preds = model.predict(temp)
    adj_means[g] = preds.mean()

print('Adjusted means:', adj_means)

# Compute differences: Homo sapiens vs each non-human genus
homo_mean = adj_means['Homo sapiens']
for g in genera:
    if g != 'Homo sapiens':
        print('Homo sapiens -', g, 'difference', homo_mean - adj_means[g])

# Use linear hypothesis test for each non-human genus vs Homo
# We can relevel genus to Homo sapiens for direct coefficients

df['genus'] = df['genus'].astype('category')
# Ensure Homo sapiens is reference
if 'Homo sapiens' in df['genus'].cat.categories:
    df['genus'] = df['genus'].cat.reorder_categories(['Homo sapiens'] + [g for g in df['genus'].cat.categories if g != 'Homo sapiens'])

model_homo_ref = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['specimen']}
)

print(model_homo_ref.summary())

# Extract coefficients for each non-human genus vs Homo
params = model_homo_ref.params
pvalues = model_homo_ref.pvalues

for term in params.index:
    if term.startswith('C(genus)'):
        print(term, 'coef', params[term], 'p', pvalues[term])
