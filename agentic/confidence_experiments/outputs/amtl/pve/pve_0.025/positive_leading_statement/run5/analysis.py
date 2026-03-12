import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
path = 'amtl.csv'
df = pd.read_csv(path)

# Derived variables
# frequency proxy: num_amtl per socket (note num_amtl is noisy and can be negative)
df['amtl_rate'] = df['num_amtl'] / df['sockets']
df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Model 1: human vs non-human, adjust for age, sex, tooth class
model1 = smf.ols('amtl_rate ~ is_human + age + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')

# Model 2: genus categories, adjust for age, sex, tooth class
model2 = smf.ols('amtl_rate ~ C(genus) + age + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')

# Summaries
print('Model 1 (human vs non-human)')
print(model1.summary())

print('\nModel 2 (genus categories)')
print(model2.summary())

# Effect size and CI for is_human
coef = model1.params['is_human']
se = model1.bse['is_human']
ci_low = coef - 1.96 * se
ci_high = coef + 1.96 * se
pval = model1.pvalues['is_human']
print('\nHuman effect (amtl_rate): coef', coef, 'SE', se, '95% CI', (ci_low, ci_high), 'p', pval)

# Provide descriptive means (unadjusted)
mean_human = df.loc[df['is_human'] == 1, 'amtl_rate'].mean()
mean_nonhuman = df.loc[df['is_human'] == 0, 'amtl_rate'].mean()
print('\nUnadjusted mean amtl_rate: human', mean_human, 'non-human', mean_nonhuman, 'diff', mean_human - mean_nonhuman)

# Alternative on raw num_amtl
model1_raw = smf.ols('num_amtl ~ is_human + age + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')
coef_raw = model1_raw.params['is_human']
se_raw = model1_raw.bse['is_human']
ci_raw = (coef_raw - 1.96*se_raw, coef_raw + 1.96*se_raw)
print('\nModel 1 raw num_amtl (human vs non-human)')
print(model1_raw.summary())
print('\nHuman effect (num_amtl): coef', coef_raw, 'SE', se_raw, '95% CI', ci_raw, 'p', model1_raw.pvalues['is_human'])

# Save key stats for later use in reasoning
stats = {
    'coef_rate': coef,
    'se_rate': se,
    'ci_low_rate': ci_low,
    'ci_high_rate': ci_high,
    'p_rate': pval,
    'mean_rate_human': mean_human,
    'mean_rate_nonhuman': mean_nonhuman,
    'coef_raw': coef_raw,
    'p_raw': model1_raw.pvalues['is_human'],
}
print('\nKEY_STATS', stats)
