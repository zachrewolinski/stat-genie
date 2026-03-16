import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

# Load dataset

df = pd.read_csv('amtl.csv')

# Ensure categories

df['feature1'] = df['feature1'].astype('category')
df['feature8'] = df['feature8'].astype('category')

# Fit OLS model with Homo sapiens as reference
formula = 'feature3 ~ C(feature8, Treatment(reference="Homo sapiens")) + C(feature1) + feature5 + feature7'
model = smf.ols(formula, data=df).fit(cov_type='HC3')

print(model.summary())

# Extract genus coefficients
params = model.params
pvalues = model.pvalues

for genus in ['Pan', 'Papio', 'Pongo']:
    term = f'C(feature8, Treatment(reference="Homo sapiens"))[T.{genus}]'
    coef = params.get(term, np.nan)
    pval = pvalues.get(term, np.nan)
    print(f'{genus} vs Homo sapiens coef (genus - Homo): {coef:.4f}, p={pval:.4g}')

# Compute adjusted means (marginal means) by genus
adj_means = {}
for genus in df['feature8'].cat.categories:
    df_tmp = df.copy()
    df_tmp['feature8'] = genus
    preds = model.predict(df_tmp)
    adj_means[genus] = preds.mean()

print('Adjusted means (model-based):')
for genus, mean in adj_means.items():
    print(f'  {genus}: {mean:.4f}')

# Difference in adjusted means vs Homo
homo_mean = adj_means.get('Homo sapiens')
print('Adjusted mean differences (Homo - other):')
for genus, mean in adj_means.items():
    if genus == 'Homo sapiens':
        continue
    print(f'  Homo - {genus}: {homo_mean - mean:.4f}')

# Omnibus test for genus effect
anova = anova_lm(model, typ=2)
print('ANOVA (type II):')
print(anova)
