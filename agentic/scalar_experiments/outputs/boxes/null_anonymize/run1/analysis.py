import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('boxes.csv')
# Rename for clarity
_df = _df.rename(columns={
    'feature1': 'outcome',
    'feature2': 'gender',
    'feature3': 'age',
    'feature4': 'majority_first',
    'feature5': 'site'
})

# Outcome encoding
# 1 = unchosen, 2 = majority, 3 = minority
_df['social_choice'] = _df['outcome'].isin([2,3]).astype(int)  # reliance on social info
_df['majority_choice'] = (_df['outcome'] == 2).astype(int)
_df['minority_choice'] = (_df['outcome'] == 3).astype(int)

# Basic rates
rates_by_site = _df.groupby('site').agg(
    n=('outcome','size'),
    social_rate=('social_choice','mean'),
    majority_rate=('majority_choice','mean'),
    minority_rate=('minority_choice','mean')
).reset_index()

# Age trend: treat age as numeric
# Logistic regression for social_choice ~ age + C(site)
model_social = smf.glm('social_choice ~ age + C(site)', data=_df, family=sm.families.Binomial()).fit()

# Logistic regression for majority preference among social choices only
_df_social = _df[_df['social_choice'] == 1].copy()
model_majority = smf.glm('majority_choice ~ age + C(site)', data=_df_social, family=sm.families.Binomial()).fit()

# Likelihood ratio tests for site effect and age effect
# Compare full vs reduced models
model_social_no_site = smf.glm('social_choice ~ age', data=_df, family=sm.families.Binomial()).fit()
model_social_no_age = smf.glm('social_choice ~ C(site)', data=_df, family=sm.families.Binomial()).fit()

model_majority_no_site = smf.glm('majority_choice ~ age', data=_df_social, family=sm.families.Binomial()).fit()
model_majority_no_age = smf.glm('majority_choice ~ C(site)', data=_df_social, family=sm.families.Binomial()).fit()


def lr_test(full, reduced):
    lr = 2 * (full.llf - reduced.llf)
    df = full.df_model - reduced.df_model
    p = sm.stats.chisqprob(lr, df) if hasattr(sm.stats, 'chisqprob') else sm.stats.chisqprob(lr, df)
    return lr, df, p

# statsmodels may not have chisqprob in newer versions; use scipy
from scipy import stats

def lr_test_sc(full, reduced):
    lr = 2 * (full.llf - reduced.llf)
    df = full.df_model - reduced.df_model
    p = stats.chi2.sf(lr, df)
    return lr, df, p

social_site_lr = lr_test_sc(model_social, model_social_no_site)
social_age_lr = lr_test_sc(model_social, model_social_no_age)

majority_site_lr = lr_test_sc(model_majority, model_majority_no_site)
majority_age_lr = lr_test_sc(model_majority, model_majority_no_age)

# Effect sizes: range across sites
social_range = rates_by_site['social_rate'].max() - rates_by_site['social_rate'].min()
majority_range = rates_by_site['majority_rate'].max() - rates_by_site['majority_rate'].min()

# Age correlation (point-biserial) for social and majority
social_age_corr = np.corrcoef(_df['age'], _df['social_choice'])[0,1]
majority_age_corr = np.corrcoef(_df_social['age'], _df_social['majority_choice'])[0,1]

# Output key stats
print('N', len(_df))
print('Sites', rates_by_site.shape[0])
print('Social rate overall', _df['social_choice'].mean())
print('Majority rate overall', _df['majority_choice'].mean())
print('Minority rate overall', _df['minority_choice'].mean())
print('\nRates by site')
print(rates_by_site)
print('\nSocial choice model (age + site)')
print(model_social.summary())
print('\nMajority choice model among social (age + site)')
print(model_majority.summary())

print('\nLR tests (social_choice)')
print('site effect', social_site_lr)
print('age effect', social_age_lr)

print('\nLR tests (majority_choice among social)')
print('site effect', majority_site_lr)
print('age effect', majority_age_lr)

print('\nEffect sizes')
print('social_range', social_range)
print('majority_range', majority_range)
print('social_age_corr', social_age_corr)
print('majority_age_corr', majority_age_corr)
