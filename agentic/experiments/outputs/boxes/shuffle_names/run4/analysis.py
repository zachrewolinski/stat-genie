import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

# Load data
# Columns:
# majority_first: 1=unchosen, 2=majority, 3=minority
# y: site id (1-8), used as culture proxy

df = pd.read_csv('boxes.csv')

# Outcomes
# Reliance on social information: chose demonstrated option (majority or minority)
# Preference for majority cues: chose majority option

df['social_choice'] = df['majority_first'].isin([2, 3]).astype(int)
df['majority_choice'] = (df['majority_first'] == 2).astype(int)

# Logistic regression with age (developmental stage) and site (culture proxy)
model_majority = smf.logit('majority_choice ~ age + C(y)', data=df).fit(disp=False)
model_social = smf.logit('social_choice ~ age + C(y)', data=df).fit(disp=False)

# Likelihood ratio tests for site effects (culture differences)
model_majority_nosite = smf.logit('majority_choice ~ age', data=df).fit(disp=False)
model_social_nosite = smf.logit('social_choice ~ age', data=df).fit(disp=False)

lr_majority = 2 * (model_majority.llf - model_majority_nosite.llf)
lr_social = 2 * (model_social.llf - model_social_nosite.llf)

df_site = model_majority.df_model - model_majority_nosite.df_model
p_majority_site = stats.chi2.sf(lr_majority, df_site)
p_social_site = stats.chi2.sf(lr_social, df_site)

# Age-group chi-square tests
bins = [3, 6, 9, 12, 14]
labels = ['4-6', '7-9', '10-12', '13-14']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

ct_majority = pd.crosstab(df['age_group'], df['majority_choice'])
ct_social = pd.crosstab(df['age_group'], df['social_choice'])

chi_majority = stats.chi2_contingency(ct_majority)
chi_social = stats.chi2_contingency(ct_social)

# Descriptive rates
site_majority = df.groupby('y')['majority_choice'].mean()
site_social = df.groupby('y')['social_choice'].mean()

age_majority = df.groupby('age_group')['majority_choice'].mean()
age_social = df.groupby('age_group')['social_choice'].mean()

# Print results for review
print('Logit majority_choice ~ age + C(y)')
print(model_majority.summary())
print('\nLogit social_choice ~ age + C(y)')
print(model_social.summary())

print('\nLR test for site effect (majority choice): chi2=%.3f df=%d p=%.4f' % (lr_majority, df_site, p_majority_site))
print('LR test for site effect (social choice): chi2=%.3f df=%d p=%.4f' % (lr_social, df_site, p_social_site))

print('\nChi-square age group vs majority choice: chi2=%.3f df=%d p=%.4f' % (chi_majority[0], chi_majority[2], chi_majority[1]))
print('Chi-square age group vs social choice: chi2=%.3f df=%d p=%.4f' % (chi_social[0], chi_social[2], chi_social[1]))

print('\nSite-level majority choice rates:')
print(site_majority)
print('\nSite-level social choice rates:')
print(site_social)

print('\nAge-group majority choice rates:')
print(age_majority)
print('\nAge-group social choice rates:')
print(age_social)
