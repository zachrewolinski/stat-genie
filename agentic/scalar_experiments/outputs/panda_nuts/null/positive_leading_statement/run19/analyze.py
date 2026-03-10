import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


df = pd.read_csv('panda_nuts.csv')

# Clean / derive efficiency
# avoid division by zero

df = df.copy()

# Ensure numeric
for col in ['nuts_opened', 'seconds', 'age']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Efficiency as nuts per second

df['efficiency'] = df['nuts_opened'] / df['seconds']

# Drop rows with missing or nonpositive seconds

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna(subset=['efficiency', 'age', 'sex', 'help'])

df = df[df['seconds'] > 0]

# Encode categorical variables

df['sex'] = df['sex'].astype('category')
df['help'] = df['help'].astype('category')

df['hammer'] = df['hammer'].astype('category')

print('Rows:', len(df))
print(df[['efficiency','age','sex','help','hammer']].head())

# Descriptive stats by sex and help
print('\nEfficiency by sex:')
print(df.groupby('sex')['efficiency'].describe())

print('\nEfficiency by help:')
print(df.groupby('help')['efficiency'].describe())

# Nonparametric tests (efficiency is skewed)
sex_groups = [g['efficiency'].values for _, g in df.groupby('sex')]
help_groups = [g['efficiency'].values for _, g in df.groupby('help')]

mw_sex = stats.mannwhitneyu(sex_groups[0], sex_groups[1], alternative='two-sided')
mw_help = stats.mannwhitneyu(help_groups[0], help_groups[1], alternative='two-sided')
spearman_age = stats.spearmanr(df['age'], df['efficiency'])

print('\nMann-Whitney U (sex):', mw_sex)
print('Mann-Whitney U (help):', mw_help)
print('Spearman (age vs efficiency):', spearman_age)

# OLS regression: efficiency ~ age + sex + help
model1 = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df).fit(cov_type='HC3')
print('\nModel 1 (no hammer)')
print(model1.summary())

# OLS regression with hammer control
model2 = smf.ols('efficiency ~ age + C(sex) + C(help) + C(hammer)', data=df).fit(cov_type='HC3')
print('\nModel 2 (with hammer)')
print(model2.summary())

# Alternative: Poisson with offset log(seconds) for counts
# Avoid zero seconds done already
poisson = smf.glm('nuts_opened ~ age + C(sex) + C(help) + C(hammer)', data=df,
                  family=sm.families.Poisson(), offset=np.log(df['seconds'])).fit(cov_type='HC3')
print('\nPoisson with offset')
print(poisson.summary())

# Save key results for later
results = {
    'n': len(df),
    'mw_sex_U': float(mw_sex.statistic),
    'mw_sex_p': float(mw_sex.pvalue),
    'mw_help_U': float(mw_help.statistic),
    'mw_help_p': float(mw_help.pvalue),
    'spearman_age_r': float(spearman_age.correlation),
    'spearman_age_p': float(spearman_age.pvalue),
    'model1_params': model1.params.to_dict(),
    'model1_pvalues': model1.pvalues.to_dict(),
    'model2_params': model2.params.to_dict(),
    'model2_pvalues': model2.pvalues.to_dict(),
    'poisson_params': poisson.params.to_dict(),
    'poisson_pvalues': poisson.pvalues.to_dict(),
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)
