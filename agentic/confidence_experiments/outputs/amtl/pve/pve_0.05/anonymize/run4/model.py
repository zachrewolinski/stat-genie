import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('amtl.csv')

# Create AMTL rate (missing per observable sockets)
df = df.copy()
df['amtl_rate'] = df['feature3'] / df['feature4']

# Create human indicator
df['human'] = (df['feature8'] == 'Homo sapiens').astype(int)

# Encode tooth class as categorical
df['tooth_class'] = df['feature1'].astype('category')

# Center age and sex for stability
df['age_c'] = df['feature5'] - df['feature5'].mean()
df['sex_c'] = df['feature7'] - df['feature7'].mean()

# OLS with robust SEs
model = smf.ols('amtl_rate ~ human + age_c + sex_c + C(tooth_class)', data=df).fit(cov_type='HC3')

# Extract human coefficient
coef = model.params['human']
se = model.bse['human']
pval = model.pvalues['human']
ci_low, ci_high = model.conf_int().loc['human']

print('N', len(df))
print('Human coef', coef)
print('SE', se)
print('pval', pval)
print('95% CI', (ci_low, ci_high))
print(model.summary().as_text())

# Also run model with genus categories to compare
model_genus = smf.ols('amtl_rate ~ C(feature8) + age_c + sex_c + C(tooth_class)', data=df).fit(cov_type='HC3')
print(model_genus.summary().as_text())

# Calculate mean rates by genus (raw)
means = df.groupby('feature8')['amtl_rate'].mean().sort_values()
print('Mean amtl_rate by genus:')
print(means)
