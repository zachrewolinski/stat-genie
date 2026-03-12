import pandas as pd
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('amtl.csv')

# Map columns to semantic names based on metadata/value patterns
df['genus_name'] = df['tooth_class']  # Homo/Pan/Papio/Pongo
df['tooth_class'] = df['sockets']  # Anterior/Posterior/Premolar
df['age_at_death'] = df['pop']
df['prob_male'] = df['stdev_age']
df['num_missing'] = df['num_amtl']
df['n_sockets'] = df['age']

# AMTL rate
df['amtl_rate'] = df['num_missing'] / df['n_sockets']

# Human indicator
df['is_human'] = (df['genus_name'] == 'Homo sapiens').astype(int)

# Model: AMTL rate ~ human + age + sex + tooth class
model = smf.ols('amtl_rate ~ is_human + age_at_death + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')

coef = model.params['is_human']
pval = model.pvalues['is_human']
ci_low, ci_high = model.conf_int().loc['is_human']

# Adjusted mean difference via counterfactual predictions
df_non = df.copy()
df_non['is_human'] = 0
df_hum = df.copy()
df_hum['is_human'] = 1
pred_diff = (model.predict(df_hum) - model.predict(df_non)).mean()

# Unadjusted means for context
mean_rate_by_genus = df.groupby('genus_name')['amtl_rate'].mean().to_dict()

print('coef_is_human', coef)
print('pval_is_human', pval)
print('ci_is_human', (ci_low, ci_high))
print('pred_diff', pred_diff)
print('mean_rate_by_genus', mean_rate_by_genus)
print('rate_over_1_fraction', (df['amtl_rate'] > 1).mean())
