import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
df = pd.read_csv('boxes.csv')

# Recode outcomes
# feature1: 1=unchosen, 2=majority, 3=minority
# Social reliance: chose any demonstrated option (majority or minority)
df['social'] = df['feature1'].isin([2, 3]).astype(int)
# Majority preference among social choices
df['majority'] = (df['feature1'] == 2).astype(int)

# Helper: likelihood ratio test between nested models

def lr_test(reduced, full):
    lr_stat = 2 * (full.llf - reduced.llf)
    df_diff = full.df_model - reduced.df_model
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return lr_stat, df_diff, p_value

results = {}

# Model 1: Social reliance (binary)
# Base: age only
m_social_age = smf.logit('social ~ feature3', data=df).fit(disp=0)
# Add culture/site
m_social_age_site = smf.logit('social ~ feature3 + C(feature5)', data=df).fit(disp=0)
# Add interaction age x site
m_social_age_site_int = smf.logit('social ~ feature3 * C(feature5)', data=df).fit(disp=0)

results['social_age_coef'] = m_social_age_site.params['feature3']
results['social_age_p'] = m_social_age_site.pvalues['feature3']

lr_site_social = lr_test(m_social_age, m_social_age_site)
lr_int_social = lr_test(m_social_age_site, m_social_age_site_int)
results['social_site_lr'] = lr_site_social
results['social_int_lr'] = lr_int_social

# Model 2: Majority preference among social choices
social_df = df[df['social'] == 1].copy()

m_maj_age = smf.logit('majority ~ feature3', data=social_df).fit(disp=0)
m_maj_age_site = smf.logit('majority ~ feature3 + C(feature5)', data=social_df).fit(disp=0)
m_maj_age_site_int = smf.logit('majority ~ feature3 * C(feature5)', data=social_df).fit(disp=0)

results['maj_age_coef'] = m_maj_age_site.params['feature3']
results['maj_age_p'] = m_maj_age_site.pvalues['feature3']

lr_site_maj = lr_test(m_maj_age, m_maj_age_site)
lr_int_maj = lr_test(m_maj_age_site, m_maj_age_site_int)
results['maj_site_lr'] = lr_site_maj
results['maj_int_lr'] = lr_int_maj

# Descriptives
site_summary = df.groupby('feature5').agg(
    n=('feature1','size'),
    social_rate=('social','mean'),
    majority_rate=('majority','mean')
).reset_index()

age_bins = pd.cut(df['feature3'], bins=[3.5,6.5,9.5,12.5,14.5], labels=['4-6','7-9','10-12','13-14'])
age_summary = df.assign(age_bin=age_bins).groupby('age_bin').agg(
    n=('feature1','size'),
    social_rate=('social','mean'),
    majority_rate=('majority','mean')
).reset_index()

# Save outputs for quick inspection
site_summary.to_csv('site_summary.csv', index=False)
age_summary.to_csv('age_summary.csv', index=False)

print('Social reliance model:')
print('Age coef (with site):', results['social_age_coef'], 'p=', results['social_age_p'])
print('LR test site effect (age vs age+site):', results['social_site_lr'])
print('LR test age*site interaction:', results['social_int_lr'])

print('\nMajority preference model (social choices only):')
print('Age coef (with site):', results['maj_age_coef'], 'p=', results['maj_age_p'])
print('LR test site effect (age vs age+site):', results['maj_site_lr'])
print('LR test age*site interaction:', results['maj_int_lr'])

print('\nSite summary:\n', site_summary)
print('\nAge summary:\n', age_summary)
