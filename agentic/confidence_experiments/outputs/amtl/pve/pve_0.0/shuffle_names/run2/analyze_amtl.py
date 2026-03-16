import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('amtl.csv')

# Map columns to semantic names
df['genus_name'] = df['tooth_class']  # Homo/Pan/Papio/Pongo
df['tooth_class'] = df['sockets']  # anterior/posterior/premolar
df['age_at_death'] = df['pop']
df['prob_male'] = df['stdev_age']
df['num_missing'] = df['num_amtl']
df['n_sockets'] = df['age']

# Compute rate
df['amtl_rate'] = df['num_missing'] / df['n_sockets']

print('Rate > 1 fraction:', (df['amtl_rate'] > 1).mean())

# Create human indicator
df['is_human'] = (df['genus_name'] == 'Homo sapiens').astype(int)

# OLS on rate with robust SE
model = smf.ols('amtl_rate ~ is_human + age_at_death + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')
print(model.summary())

# Extract coefficient
coef = model.params['is_human']
pval = model.pvalues['is_human']

# Compute adjusted mean difference using predictions
df_non = df.copy()
df_non['is_human'] = 0
df_hum = df.copy()
df_hum['is_human'] = 1
pred_diff = (model.predict(df_hum) - model.predict(df_non)).mean()

print('is_human coef:', coef)
print('pval:', pval)
print('predicted mean diff:', pred_diff)
