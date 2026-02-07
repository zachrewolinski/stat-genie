import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2

# Load data

df = pd.read_csv('boxes.csv')

# Derived outcomes
# social_choice: chose a demonstrated option (majority or minority)
df['social_choice'] = df['majority_first'].isin([2, 3]).astype(int)
# majority_choice among demonstrated choices only (exclude unchosen)
df_dem = df[df['majority_first'].isin([2, 3])].copy()
df_dem['majority_choice'] = (df_dem['majority_first'] == 2).astype(int)

# Logistic regression for social_choice ~ age + site (y)
model_social = smf.glm('social_choice ~ age + C(y)', data=df, family=sm.families.Binomial()).fit()
model_social_null = smf.glm('social_choice ~ 1', data=df, family=sm.families.Binomial()).fit()

# Likelihood ratio test for full model vs null
lr_social = 2 * (model_social.llf - model_social_null.llf)
df_social = model_social.df_model - model_social_null.df_model
p_social = chi2.sf(lr_social, df_social)

# Test age effect alone (compare to model without age)
model_social_no_age = smf.glm('social_choice ~ C(y)', data=df, family=sm.families.Binomial()).fit()
lr_social_age = 2 * (model_social.llf - model_social_no_age.llf)
df_social_age = model_social.df_model - model_social_no_age.df_model
p_social_age = chi2.sf(lr_social_age, df_social_age)

# Test site effect (compare to model without site)
model_social_no_site = smf.glm('social_choice ~ age', data=df, family=sm.families.Binomial()).fit()
lr_social_site = 2 * (model_social.llf - model_social_no_site.llf)
df_social_site = model_social.df_model - model_social_no_site.df_model
p_social_site = chi2.sf(lr_social_site, df_social_site)

# Logistic regression for majority_choice among demonstrated
model_major = smf.glm('majority_choice ~ age + C(y)', data=df_dem, family=sm.families.Binomial()).fit()
model_major_null = smf.glm('majority_choice ~ 1', data=df_dem, family=sm.families.Binomial()).fit()

lr_major = 2 * (model_major.llf - model_major_null.llf)
df_major = model_major.df_model - model_major_null.df_model
p_major = chi2.sf(lr_major, df_major)

# Age effect in majority preference
model_major_no_age = smf.glm('majority_choice ~ C(y)', data=df_dem, family=sm.families.Binomial()).fit()
lr_major_age = 2 * (model_major.llf - model_major_no_age.llf)
df_major_age = model_major.df_model - model_major_no_age.df_model
p_major_age = chi2.sf(lr_major_age, df_major_age)

# Site effect in majority preference
model_major_no_site = smf.glm('majority_choice ~ age', data=df_dem, family=sm.families.Binomial()).fit()
lr_major_site = 2 * (model_major.llf - model_major_no_site.llf)
df_major_site = model_major.df_model - model_major_no_site.df_model
p_major_site = chi2.sf(lr_major_site, df_major_site)

# Summaries
print('Social choice rate:', df['social_choice'].mean())
print('Majority choice rate among demonstrated:', df_dem['majority_choice'].mean())
print('Social choice model LR p:', p_social)
print('Social choice age effect p:', p_social_age)
print('Social choice site effect p:', p_social_site)
print('Majority choice model LR p:', p_major)
print('Majority choice age effect p:', p_major_age)
print('Majority choice site effect p:', p_major_site)

# Effect sizes (odds ratios) for age
print('Social choice age OR:', np.exp(model_social.params['age']))
print('Majority choice age OR:', np.exp(model_major.params['age']))

# Site-level rates
site_social = df.groupby('y')['social_choice'].mean()
site_major = df_dem.groupby('y')['majority_choice'].mean()
print('Site social choice rates:', site_social.to_dict())
print('Site majority choice rates:', site_major.to_dict())

# Age-group rates
age_social = df.groupby('age')['social_choice'].mean()
age_major = df_dem.groupby('age')['majority_choice'].mean()
print('Age social choice rates:', age_social.to_dict())
print('Age majority choice rates:', age_major.to_dict())
