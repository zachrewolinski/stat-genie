import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data

df = pd.read_csv('boxes.csv')

# Outcomes
# majority option = 2, minority option = 3, unchosen = 1

df['choose_majority'] = (df['majority_first'] == 2).astype(int)
df['choose_social'] = df['majority_first'].isin([2, 3]).astype(int)

# Site as categorical

df['site'] = df['y'].astype('category')

# Helper: likelihood ratio test for nested models

def lr_test(full_model, reduced_model):
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df_diff = full_model.df_model - reduced_model.df_model
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return lr_stat, df_diff, p_value

results = {}

# Majority preference model

full_maj = smf.logit('choose_majority ~ age + C(site)', data=df).fit(disp=False)
red_maj_site = smf.logit('choose_majority ~ age', data=df).fit(disp=False)
red_maj_age = smf.logit('choose_majority ~ C(site)', data=df).fit(disp=False)

lr_site_maj = lr_test(full_maj, red_maj_site)
lr_age_maj = lr_test(full_maj, red_maj_age)

# Social reliance model (choose demonstrated options)

full_soc = smf.logit('choose_social ~ age + C(site)', data=df).fit(disp=False)
red_soc_site = smf.logit('choose_social ~ age', data=df).fit(disp=False)
red_soc_age = smf.logit('choose_social ~ C(site)', data=df).fit(disp=False)

lr_site_soc = lr_test(full_soc, red_soc_site)
lr_age_soc = lr_test(full_soc, red_soc_age)

# Descriptive: age bins

bins = [3, 6, 9, 14]
labels = ['4-6', '7-9', '10-14']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

summary = df.groupby(['age_group', 'site']).agg(
    n=('majority_first', 'size'),
    majority_rate=('choose_majority', 'mean'),
    social_rate=('choose_social', 'mean'),
).reset_index()

# Overall by age group

overall_age = df.groupby('age_group').agg(
    n=('majority_first', 'size'),
    majority_rate=('choose_majority', 'mean'),
    social_rate=('choose_social', 'mean'),
).reset_index()

# Overall by site

overall_site = df.groupby('site').agg(
    n=('majority_first', 'size'),
    majority_rate=('choose_majority', 'mean'),
    social_rate=('choose_social', 'mean'),
).reset_index()

# Print key results
print('Majority preference model:')
print(full_maj.summary())
print('\nLR test site effect (age vs age+site):', lr_site_maj)
print('LR test age effect (site vs site+age):', lr_age_maj)

print('\nSocial reliance model:')
print(full_soc.summary())
print('\nLR test site effect (age vs age+site):', lr_site_soc)
print('LR test age effect (site vs site+age):', lr_age_soc)

print('\nOverall by age group:')
print(overall_age)
print('\nOverall by site:')
print(overall_site)
print('\nBy age group and site (first 10 rows):')
print(summary.head(10))
