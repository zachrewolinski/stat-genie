import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('amtl.csv')

# Basic subset
sub = df[['num_amtl','age','prob_male','tooth_class','genus']].dropna().copy()

sub['is_human'] = (sub['genus'] == 'Homo sapiens').astype(int)

# Model 1: human vs non-human
model_human = smf.ols('num_amtl ~ is_human + age + prob_male + C(tooth_class)', data=sub).fit()

# Model 2: genus-specific (Homo as reference)
model_genus = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=sub).fit()

# Summaries
print('Model human vs non-human')
print(model_human.summary())
print('\nModel genus-specific')
print(model_genus.summary())

# Extract key coefficients
coef_human = model_human.params['is_human']
pval_human = model_human.pvalues['is_human']
ci_human = model_human.conf_int().loc['is_human'].tolist()

print('\nHuman indicator coef', coef_human, 'p', pval_human, 'CI', ci_human)

# Genus-specific coefficients for non-human relative to human
for genus in ['Pan','Papio','Pongo']:
    term = f'C(genus)[T.{genus}]'
    print(genus, 'coef', model_genus.params[term], 'p', model_genus.pvalues[term], 'CI', model_genus.conf_int().loc[term].tolist())

# Means (unadjusted) for reference
print('\nUnadjusted mean num_amtl by genus')
print(sub.groupby('genus')['num_amtl'].mean())

