import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2

# Load data
path = 'boxes.csv'
df = pd.read_csv(path)

# Recode outcomes
# majority_first: 1=unchosen, 2=majority, 3=minority

df['social_reliance'] = (df['majority_first'] != 1).astype(int)
# majority preference conditional on using social info
cond = df['majority_first'].isin([2,3])
df['majority_choice'] = np.where(cond, (df['majority_first'] == 2).astype(int), np.nan)

# Treat site id as culture/site
# y: site ID 1-8

df['site'] = df['y'].astype('category')

# Age groups for descriptive summary
bins = [4,6,9,12,14]
labels = ['4-6','7-9','10-12','13-14']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, include_lowest=True, right=True)

# Descriptive summaries
summary_social = df.groupby(['site','age_group'])['social_reliance'].mean().unstack()
summary_majority = df[cond].groupby(['site','age_group'])['majority_choice'].mean().unstack()

print('Social reliance (P choose demonstrated option) by site x age group:')
print(summary_social)
print('\nMajority preference (P choose majority | chose demonstrated) by site x age group:')
print(summary_majority)

# Logistic regression: social reliance ~ age + site
model_social = smf.glm('social_reliance ~ age + C(site)', data=df, family=sm.families.Binomial()).fit()
print('\nGLM social reliance ~ age + site')
print(model_social.summary())

# Logistic regression: majority preference ~ age + site (conditional)
df_cond = df[cond].copy()
model_majority = smf.glm('majority_choice ~ age + C(site)', data=df_cond, family=sm.families.Binomial()).fit()
print('\nGLM majority preference ~ age + site (conditional on demonstrated choice)')
print(model_majority.summary())

# Add age x site interaction to test variation in developmental stages across cultures
model_social_int = smf.glm('social_reliance ~ age * C(site)', data=df, family=sm.families.Binomial()).fit()
print('\nGLM social reliance ~ age * site')
print(model_social_int.summary())

model_majority_int = smf.glm('majority_choice ~ age * C(site)', data=df_cond, family=sm.families.Binomial()).fit()
print('\nGLM majority preference ~ age * site (conditional)')
print(model_majority_int.summary())

# Likelihood ratio tests for interaction
lr_social = 2*(model_social_int.llf - model_social.llf)
# df difference = number of added params
lr_df_social = model_social_int.df_model - model_social.df_model
p_social = chi2.sf(lr_social, lr_df_social)

lr_majority = 2*(model_majority_int.llf - model_majority.llf)
lr_df_majority = model_majority_int.df_model - model_majority.df_model
p_majority = chi2.sf(lr_majority, lr_df_majority)

print('\nLR test interaction (social reliance): chi2=%.3f, df=%d, p=%.4f' % (lr_social, lr_df_social, p_social))
print('LR test interaction (majority preference): chi2=%.3f, df=%d, p=%.4f' % (lr_majority, lr_df_majority, p_majority))

# Overall effect of age and site from base models
print('\nAge effect social reliance coef, p-value:', model_social.params['age'], model_social.pvalues['age'])
print('Age effect majority preference coef, p-value:', model_majority.params['age'], model_majority.pvalues['age'])

# Site effect likelihood ratio (overall)
# Compare to age-only model
model_social_age = smf.glm('social_reliance ~ age', data=df, family=sm.families.Binomial()).fit()
model_majority_age = smf.glm('majority_choice ~ age', data=df_cond, family=sm.families.Binomial()).fit()

lr_site_social = 2*(model_social.llf - model_social_age.llf)
lr_df_site_social = model_social.df_model - model_social_age.df_model
p_site_social = chi2.sf(lr_site_social, lr_df_site_social)

lr_site_majority = 2*(model_majority.llf - model_majority_age.llf)
lr_df_site_majority = model_majority.df_model - model_majority_age.df_model
p_site_majority = chi2.sf(lr_site_majority, lr_df_site_majority)

print('\nLR test site effect (social reliance): chi2=%.3f, df=%d, p=%.4f' % (lr_site_social, lr_df_site_social, p_site_social))
print('LR test site effect (majority preference): chi2=%.3f, df=%d, p=%.4f' % (lr_site_majority, lr_df_site_majority, p_site_majority))
