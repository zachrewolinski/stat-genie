import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


df = pd.read_csv('amtl.csv')

# Ensure categorical
for col in ['genus', 'tooth_class', 'specimen']:
    df[col] = df[col].astype('category')

# Use Homo sapiens as reference category
# statsmodels uses alphabetical by default; set category order explicitly
if 'Homo sapiens' in df['genus'].cat.categories:
    new_order = ['Homo sapiens'] + [g for g in df['genus'].cat.categories if g != 'Homo sapiens']
    df['genus'] = df['genus'].cat.reorder_categories(new_order)

# Build linear model controlling for age, sex (prob_male), tooth_class
# Cluster-robust SE by specimen since each specimen appears multiple times
formula = 'num_amtl ~ C(genus) + age + prob_male + C(tooth_class)'
model = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['specimen']})

# Extract coefficients for genus comparisons vs Homo sapiens
params = model.params
pvalues = model.pvalues

# Identify genus terms
genus_terms = [term for term in params.index if term.startswith('C(genus)')]

results = {}
for term in genus_terms:
    genus_name = term.split('T.')[-1].rstrip(']')
    results[genus_name] = {
        'coef': params[term],
        'pvalue': pvalues[term]
    }

# Compute model-based adjusted means by genus (standardization)
# For each genus, set genus to that level and average predictions over observed covariates
adjusted_means = {}
for genus in df['genus'].cat.categories:
    tmp = df.copy()
    tmp['genus'] = genus
    adjusted_means[genus] = float(model.predict(tmp).mean())

# Compute differences Homo sapiens - others
homo_mean = adjusted_means['Homo sapiens']
mean_diffs = {g: homo_mean - m for g, m in adjusted_means.items() if g != 'Homo sapiens'}

# Save analysis artifacts for inspection
analysis = {
    'n': int(df.shape[0]),
    'genus_counts': df['genus'].value_counts().to_dict(),
    'model_summary': model.summary().as_text(),
    'genus_terms': results,
    'adjusted_means': adjusted_means,
    'mean_diffs': mean_diffs,
}

with open('analysis_results.json', 'w') as f:
    json.dump(analysis, f, indent=2)

# Simple printed output
print('Genus coefficients vs Homo sapiens (cluster-robust):')
for g, info in results.items():
    print(g, 'coef', info['coef'], 'p', info['pvalue'])
print('Adjusted means:', adjusted_means)
print('Mean diffs (Homo - other):', mean_diffs)
