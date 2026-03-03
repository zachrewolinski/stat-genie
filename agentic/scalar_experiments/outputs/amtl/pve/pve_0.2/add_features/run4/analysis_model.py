import pandas as pd
import statsmodels.formula.api as smf


df = pd.read_csv('amtl.csv')

# Ensure categorical types
# Use exact string matching for genus; there might be whitespace

df['genus'] = df['genus'].astype(str)
df['tooth_class'] = df['tooth_class'].astype(str)

# Binary indicator for humans

df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Model with genus categories (Homo sapiens as reference)
model_genus = smf.ols(
    'num_amtl ~ C(genus, Treatment(reference="Homo sapiens")) + age + prob_male + C(tooth_class)',
    data=df
).fit(cov_type='HC3')

# Model with binary human vs non-human
model_human = smf.ols(
    'num_amtl ~ is_human + age + prob_male + C(tooth_class)',
    data=df
).fit(cov_type='HC3')

print('model_genus_params')
print(model_genus.params)
print('model_genus_pvalues')
print(model_genus.pvalues)

print('model_human_params')
print(model_human.params)
print('model_human_pvalues')
print(model_human.pvalues)

# Compute adjusted mean difference via is_human coefficient
print('is_human_coef', model_human.params['is_human'])
print('is_human_pvalue', model_human.pvalues['is_human'])

