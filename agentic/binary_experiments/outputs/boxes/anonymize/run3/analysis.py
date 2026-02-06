import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('boxes.csv')

# Prepare variables
_df['site'] = _df['feature5'].astype('category')
_df['age'] = _df['feature3'].astype(float)

# Outcome 1: reliance on social information (choose majority or minority)
_df['social_choice'] = (_df['feature1'] != 1).astype(int)

# Outcome 2: majority preference among social choices
_df_social = _df[_df['feature1'].isin([2, 3])].copy()
_df_social['majority_choice'] = (_df_social['feature1'] == 2).astype(int)


def lr_test(full_model, reduced_model):
    """Likelihood-ratio test for nested models."""
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df_diff = full_model.df_model - reduced_model.df_model
    from scipy.stats import chi2
    p = chi2.sf(lr_stat, df_diff)
    return lr_stat, df_diff, p


# Model for social reliance
m_social_full = smf.logit('social_choice ~ age + C(site)', data=_df).fit(disp=False)
m_social_no_age = smf.logit('social_choice ~ C(site)', data=_df).fit(disp=False)
m_social_no_site = smf.logit('social_choice ~ age', data=_df).fit(disp=False)

social_age_lr = lr_test(m_social_full, m_social_no_age)
social_site_lr = lr_test(m_social_full, m_social_no_site)

# Model for majority preference among social choices
m_majority_full = smf.logit('majority_choice ~ age + C(site)', data=_df_social).fit(disp=False)
m_majority_no_age = smf.logit('majority_choice ~ C(site)', data=_df_social).fit(disp=False)
m_majority_no_site = smf.logit('majority_choice ~ age', data=_df_social).fit(disp=False)

majority_age_lr = lr_test(m_majority_full, m_majority_no_age)
majority_site_lr = lr_test(m_majority_full, m_majority_no_site)

# Summaries for quick inspection
print('Social reliance model (full):')
print(m_social_full.summary())
print('\nLR test age (social reliance):', social_age_lr)
print('LR test site (social reliance):', social_site_lr)

print('\nMajority preference model (full):')
print(m_majority_full.summary())
print('\nLR test age (majority preference):', majority_age_lr)
print('LR test site (majority preference):', majority_site_lr)

# Also compute simple rates by site and age bins for interpretation
_df['age_bin'] = pd.cut(_df['age'], bins=[3, 5, 7, 9, 11, 13, 15], labels=['4-5', '6-7', '8-9', '10-11', '12-13', '14'])
_df_social['age_bin'] = pd.cut(_df_social['age'], bins=[3, 5, 7, 9, 11, 13, 15], labels=['4-5', '6-7', '8-9', '10-11', '12-13', '14'])

rate_social_by_site = _df.groupby('site')['social_choice'].mean()
rate_majority_by_site = _df_social.groupby('site')['majority_choice'].mean()
rate_social_by_age = _df.groupby('age_bin')['social_choice'].mean()
rate_majority_by_age = _df_social.groupby('age_bin')['majority_choice'].mean()

print('\nSocial reliance rate by site:')
print(rate_social_by_site)
print('\nMajority preference rate by site (among social choices):')
print(rate_majority_by_site)
print('\nSocial reliance rate by age bin:')
print(rate_social_by_age)
print('\nMajority preference rate by age bin (among social choices):')
print(rate_majority_by_age)
