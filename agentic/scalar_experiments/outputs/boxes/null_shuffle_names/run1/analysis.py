import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'boxes.csv'
df = pd.read_csv(path)

# Outcome encodings
# majority_first: 1=unchosen, 2=majority, 3=minority

df['choice_majority'] = (df['majority_first'] == 2).astype(int)
df['choice_minority'] = (df['majority_first'] == 3).astype(int)
df['choice_unchosen'] = (df['majority_first'] == 1).astype(int)

df['choice_social'] = (df['majority_first'].isin([2,3])).astype(int)

# Basic rates
summary = {}
summary['n'] = len(df)
summary['majority_rate'] = df['choice_majority'].mean()
summary['minority_rate'] = df['choice_minority'].mean()
summary['unchosen_rate'] = df['choice_unchosen'].mean()
summary['social_rate'] = df['choice_social'].mean()

# Age groups
bins = [3.5, 6.5, 9.5, 12.5, 14.5]
labels = ['4-6', '7-9', '10-12', '13-14']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

age_group_rates = df.groupby('age_group', observed=True)[['choice_majority', 'choice_minority', 'choice_social']].mean()

# Site rates (y)
site_rates = df.groupby('y')[['choice_majority', 'choice_minority', 'choice_social']].mean()

# Logistic regression: majority vs others
# Predictors: age (continuous) and site (categorical)
model_majority = smf.glm('choice_majority ~ age + C(y)', data=df, family=sm.families.Binomial()).fit()

# Logistic regression: social vs unchosen
model_social = smf.glm('choice_social ~ age + C(y)', data=df, family=sm.families.Binomial()).fit()

# Likelihood ratio tests for site effect (categorical)
# Compare with reduced models without site
model_majority_age = smf.glm('choice_majority ~ age', data=df, family=sm.families.Binomial()).fit()
model_social_age = smf.glm('choice_social ~ age', data=df, family=sm.families.Binomial()).fit()

lr_majority = 2 * (model_majority.llf - model_majority_age.llf)
lr_social = 2 * (model_social.llf - model_social_age.llf)

df_sites = df['y'].nunique()
# degrees of freedom for site effects: (k-1)
df_site = df_sites - 1

from scipy import stats
p_majority_site = stats.chi2.sf(lr_majority, df_site)
p_social_site = stats.chi2.sf(lr_social, df_site)

# Extract age effect p-values
p_age_majority = model_majority.pvalues.get('age', np.nan)
p_age_social = model_social.pvalues.get('age', np.nan)

print('SUMMARY')
print(summary)
print('\nAGE GROUP RATES')
print(age_group_rates)
print('\nSITE RATES')
print(site_rates)
print('\nLOGIT: majority ~ age + C(y)')
print('age coef', model_majority.params.get('age'), 'p', p_age_majority)
print('site LR p', p_majority_site)
print('\nLOGIT: social ~ age + C(y)')
print('age coef', model_social.params.get('age'), 'p', p_age_social)
print('site LR p', p_social_site)
