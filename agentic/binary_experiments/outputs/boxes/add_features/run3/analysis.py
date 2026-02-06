import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats


def lr_test(full, reduced):
    lr_stat = 2 * (full.llf - reduced.llf)
    df = full.df_model - reduced.df_model
    p = stats.chi2.sf(lr_stat, df)
    return lr_stat, df, p


df = pd.read_csv('boxes.csv')

# Keep needed columns
needed = ['y', 'age', 'culture']
sub = df[needed].dropna().copy()

# Binary outcome: relied on social information (chose majority or minority)
sub['social'] = (sub['y'] != 1).astype(int)

# Model for social reliance
m_full = smf.logit('social ~ age + C(culture) + age:C(culture)', data=sub).fit(disp=False)
m_no_int = smf.logit('social ~ age + C(culture)', data=sub).fit(disp=False)
m_age_only = smf.logit('social ~ age', data=sub).fit(disp=False)

lr_int = lr_test(m_full, m_no_int)
lr_cult = lr_test(m_no_int, m_age_only)

# Majority preference among those choosing demonstrated options
sub2 = sub[sub['y'].isin([2, 3])].copy()
sub2['majority'] = (sub2['y'] == 2).astype(int)

m2_full = smf.logit('majority ~ age + C(culture) + age:C(culture)', data=sub2).fit(disp=False)
m2_no_int = smf.logit('majority ~ age + C(culture)', data=sub2).fit(disp=False)
m2_age_only = smf.logit('majority ~ age', data=sub2).fit(disp=False)

lr2_int = lr_test(m2_full, m2_no_int)
lr2_cult = lr_test(m2_no_int, m2_age_only)

# Descriptive rates
social_rate_by_culture = sub.groupby('culture')['social'].mean().sort_index()
majority_rate_by_culture = sub2.groupby('culture')['majority'].mean().sort_index()

# Age trend (simple correlation)
age_social_corr = sub[['age', 'social']].corr().loc['age', 'social']
age_majority_corr = sub2[['age', 'majority']].corr().loc['age', 'majority']

print('Social reliance LR tests:')
print('interaction vs no interaction:', lr_int)
print('culture effect vs age-only:', lr_cult)
print('\nMajority preference LR tests:')
print('interaction vs no interaction:', lr2_int)
print('culture effect vs age-only:', lr2_cult)

print('\nAge correlations:')
print('age vs social:', age_social_corr)
print('age vs majority (among demonstrated choices):', age_majority_corr)

print('\nSocial rate by culture:')
print(social_rate_by_culture)
print('\nMajority rate by culture (among demonstrated choices):')
print(majority_rate_by_culture)
