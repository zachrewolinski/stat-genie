import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
csv_path = 'panda_nuts.csv'
df = pd.read_csv(csv_path)

# Basic cleaning
# Normalize column names if needed (already lower)

# Compute efficiency: nuts opened per second
# Avoid division by zero
if (df['seconds'] <= 0).any():
    raise ValueError('Non-positive seconds found')

df['efficiency'] = df['nuts_opened'] / df['seconds']

# Convert categorical variables
for col in ['sex', 'help', 'hammer']:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Summary stats
summary = df[['efficiency', 'age', 'nuts_opened', 'seconds']].describe()
print('Summary:\n', summary)

print('\nCounts for sex/help:')
print(df['sex'].value_counts(dropna=False))
print(df['help'].value_counts(dropna=False))

# Model: efficiency ~ age + sex + help
model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df).fit()
print('\nOLS model summary:')
print(model.summary())

# ANOVA table for overall effects
anova_table = sm.stats.anova_lm(model, typ=2)
print('\nANOVA (type II):')
print(anova_table)

# Check for interaction? not asked but could explore
model_int = smf.ols('efficiency ~ age + C(sex) + C(help) + age:C(sex) + age:C(help)', data=df).fit()
print('\nOLS model with age interactions summary (AIC):')
print('AIC base:', model.aic, 'AIC int:', model_int.aic)

# Nonparametric tests for sex/help (Mann-Whitney)
if df['sex'].nunique() == 2:
    groups = [g['efficiency'].values for _, g in df.groupby('sex')]
    u_stat, p_val = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')
    print('\nMann-Whitney sex p=', p_val)

if df['help'].nunique() == 2:
    groups = [g['efficiency'].values for _, g in df.groupby('help')]
    u_stat, p_val = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')
    print('Mann-Whitney help p=', p_val)

# Correlation for age
corr = stats.pearsonr(df['age'], df['efficiency'])
print('\nPearson correlation age-efficiency:', corr)

# Save key results to json for later
import json
out = {
    'n': len(df),
    'model_params': model.params.to_dict(),
    'model_pvalues': model.pvalues.to_dict(),
    'model_r2': model.rsquared,
    'anova': anova_table.to_dict(),
}
with open('analysis_results.json', 'w') as f:
    json.dump(out, f, indent=2)

