import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('amtl.csv')

# Create human indicator
df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Basic OLS with human indicator
model = smf.ols('num_amtl ~ is_human + age + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')

# Full genus model with Homo sapiens as reference
df['genus'] = df['genus'].astype('category')
df['genus'] = df['genus'].cat.reorder_categories(
    ['Homo sapiens', 'Pan', 'Pongo', 'Papio'], ordered=False
)
model_genus = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')

# Compute adjusted mean difference human vs non-human using indicator
coef = model.params['is_human']
se = model.bse['is_human']
p = model.pvalues['is_human']

# Sample sizes
counts = df['genus'].value_counts()

# R-squared
r2 = model.rsquared

# Summarize effect sizes for specific genera vs human
genus_coefs = {k: v for k, v in model_genus.params.items() if k.startswith('C(genus)')}
genus_pvals = {k: v for k, v in model_genus.pvalues.items() if k.startswith('C(genus)')}

# Output concise results
print('Human indicator model:')
print(model.summary().tables[1])
print('\nR2:', r2)
print('\nCounts:', counts.to_dict())
print('\nHuman effect coef:', coef, 'SE:', se, 'p:', p)
print('\nGenus model coefficients (vs Homo sapiens):')
for k, v in genus_coefs.items():
    print(k, 'coef', v, 'p', genus_pvals[k])

