import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
amtl = pd.read_csv('amtl.csv')

# Ensure categorical
amtl['genus'] = amtl['genus'].astype('category')
amtl['tooth_class'] = amtl['tooth_class'].astype('category')

# Fit OLS with cluster-robust SEs by specimen (repeated measures across tooth classes)
model = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=amtl).fit(
    cov_type='cluster', cov_kwds={'groups': amtl['specimen']}
)

print(model.summary())

# Compute adjusted mean num_amtl by genus at observed covariates
# For each genus, set genus to that level for all rows and average predictions
adj_means = {}
for genus in amtl['genus'].cat.categories:
    temp = amtl.copy()
    temp['genus'] = genus
    adj_means[genus] = model.predict(temp).mean()

print('Adjusted means (model-based):')
for k,v in adj_means.items():
    print(k, v)

# Compute pairwise differences: Homo sapiens vs each non-human genus
# We'll use linear hypothesis testing for the model coefficients.
# Determine baseline for genus in the model
print('Baseline genus:', amtl['genus'].cat.categories[0])

# Get contrast for difference between Homo sapiens and other genus
# For treatment coding, the coefficient for C(genus)[T.X] is difference between X and baseline.
# We may need to relevel to make Homo sapiens the baseline for easier interpretation.

amtl['genus_relevel'] = amtl['genus'].cat.reorder_categories(
    ['Homo sapiens'] + [g for g in amtl['genus'].cat.categories if g != 'Homo sapiens'],
    ordered=False
)
model2 = smf.ols('num_amtl ~ C(genus_relevel) + age + prob_male + C(tooth_class)', data=amtl).fit(
    cov_type='cluster', cov_kwds={'groups': amtl['specimen']}
)
print(model2.summary())

# Extract differences (other genus vs Homo sapiens baseline)
params = model2.params
pvals = model2.pvalues
for g in [g for g in amtl['genus'].cat.categories if g != 'Homo sapiens']:
    term = f'C(genus_relevel)[T.{g}]'
    print('Diff', g, 'vs Homo sapiens:', params[term], 'p=', pvals[term])

# Save key results for downstream
res = {
    'adj_means': adj_means,
    'diffs': {g: {'coef': params[f'C(genus_relevel)[T.{g}]'], 'p': pvals[f'C(genus_relevel)[T.{g}]']} for g in [g for g in amtl['genus'].cat.categories if g != 'Homo sapiens']}
}

import json
with open('model_results.json','w') as f:
    json.dump(res, f, indent=2)
